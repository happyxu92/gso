import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import yaml

from gso.collect.execute.helpers import collect_results
from gso.collect.execute.skymgr import SkyManager
from gso.logger import logger


class DockerManager:
    """Run generated collection tasks in isolated local Docker containers."""

    cluster_prefix = "docker-gso"
    repository_dockerfile = """# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG REPO_URL
ARG REPO_NAME
RUN test -n "$REPO_URL" && test -n "$REPO_NAME" \\
    && mkdir -p /workspace \\
    && git clone --recursive "$REPO_URL" "/workspace/$REPO_NAME" \\
    && git -C "/workspace/$REPO_NAME" fetch --all --tags --prune
WORKDIR /workspace
"""
    local_repository_dockerfile = """# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG REPO_NAME
RUN test -n "$REPO_NAME" && mkdir -p /workspace
COPY repository /workspace/${REPO_NAME}
WORKDIR /workspace
"""

    def __init__(
        self,
        image: str,
        artifact_dir: Path,
        cpus: float | None = None,
        memory: str | None = None,
        platform: str | None = None,
        keep_containers: bool = False,
        repository_base_image: str | None = None,
        repository_path: Path | None = None,
        rebuild_repository_image: bool = False,
    ):
        self.image = image
        self.artifact_dir = artifact_dir
        self.cpus = cpus
        self.memory = memory
        self.platform = platform
        self.keep_containers = keep_containers
        self.repository_base_image = repository_base_image
        self.repository_path = repository_path
        self.rebuild_repository_image = rebuild_repository_image
        self._containers: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def create_workspace(
        problem, phase1_only: bool = False, test_timeout: int = 300
    ) -> Path:
        # Docker and SkyPilot consume the same generated tests and phase scripts.
        return SkyManager.create_workspace(
            problem, phase1_only=phase1_only, test_timeout=test_timeout
        )

    def _platform_args(self) -> list[str]:
        return ["--platform", self.platform] if self.platform else []

    @staticmethod
    def _run(
        command: list[str],
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            check=check,
            capture_output=capture_output,
            text=True,
        )

    @staticmethod
    def _repository(problems) -> tuple[str, str]:
        """Return the single repository shared by an execution batch."""
        repositories = {
            (problem.repo.repo_url, problem.repo.repo_name) for problem in problems
        }
        if len(repositories) != 1:
            formatted = ", ".join(
                sorted(f"{url} ({name})" for url, name in repositories)
            )
            raise ValueError(
                "Docker repository image builds require exactly one repository; "
                f"found: {formatted or 'none'}"
            )

        repo_url, repo_name = next(iter(repositories))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo_name):
            raise ValueError(f"Invalid repository name: {repo_name!r}")
        if not repo_url.strip() or any(char in repo_url for char in "\r\n"):
            raise ValueError(f"Invalid repository URL: {repo_url!r}")
        return repo_url, repo_name

    def build_repository_image(self, problems) -> None:
        """Build an execution image from a remote or local repository."""
        base_image = self.repository_base_image
        if not base_image:
            if self.repository_path is not None:
                raise ValueError("--docker-repo-path requires --docker-base-image")
            return
        if base_image == self.image:
            raise ValueError("Docker base image and repository image must differ")

        repo_url, repo_name = self._repository(problems)
        local_repo = None
        if self.repository_path is not None:
            local_repo = Path(self.repository_path).expanduser().resolve()
            if not local_repo.is_dir():
                raise ValueError(f"Docker repository path is not a directory: {local_repo}")
            if not (local_repo / ".git").is_dir():
                raise ValueError(
                    "Docker repository path must be a self-contained Git checkout "
                    f"with a .git directory: {local_repo}"
                )

        base_inspect = self._run(
            ["docker", "image", "inspect", base_image],
            check=False,
            capture_output=True,
        )
        if base_inspect.returncode != 0:
            detail = (base_inspect.stderr or base_inspect.stdout).strip()
            raise RuntimeError(
                f"Docker base image {base_image!r} is not available locally: {detail}"
            )

        build_dir = self.artifact_dir / "repository_image"
        build_dir.mkdir(parents=True, exist_ok=True)
        dockerfile = build_dir / "Dockerfile"
        dockerfile.write_text(
            self.local_repository_dockerfile
            if local_repo is not None
            else self.repository_dockerfile,
            encoding="utf-8",
        )

        command = [
            "docker",
            "build",
            *self._platform_args(),
            "--file",
            str(dockerfile),
            "--tag",
            self.image,
            "--build-arg",
            f"BASE_IMAGE={base_image}",
        ]
        if local_repo is None:
            command.extend(["--build-arg", f"REPO_URL={repo_url}"])
        command.extend(["--build-arg", f"REPO_NAME={repo_name}"])
        if self.rebuild_repository_image:
            command.append("--no-cache")

        # Stage local repositories so their .dockerignore cannot omit .git.
        temp_context = None
        if local_repo is not None:
            temp_context = tempfile.TemporaryDirectory(prefix="gso-docker-repo-")
            context_dir = Path(temp_context.name)
            shutil.copytree(local_repo, context_dir / "repository", symlinks=True)
        else:
            context_dir = build_dir
        command.append(str(context_dir))

        try:
            result = self._run(command, check=False, capture_output=True)
        finally:
            if temp_context is not None:
                temp_context.cleanup()

        build_log = build_dir / "build.log"
        build_log.write_text(
            (result.stdout or "") + (result.stderr or ""), encoding="utf-8"
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown build error").strip()
            raise RuntimeError(
                f"Failed to build Docker repository image {self.image!r}: "
                f"{detail[-4000:]}"
            )
        repository_source = local_repo if local_repo is not None else repo_url
        print(
            f"Built Docker repository image {self.image} from {base_image}; "
            f"repository={repository_source}; log={build_log}",
            flush=True,
        )

    def validate(self, problems) -> None:
        """Build when requested, then validate the image and candidate commits."""
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI not found")

        info = self._run(
            ["docker", "info", "--format", "{{.Architecture}}"],
            capture_output=True,
        )
        engine_arch = info.stdout.strip()
        self.build_repository_image(problems)

        inspect = self._run(
            [
                "docker",
                "image",
                "inspect",
                self.image,
                "--format",
                "{{.Architecture}}",
            ],
            capture_output=True,
        )
        image_arch = inspect.stdout.strip()
        normalized_arch = {
            "arm64": "arm64",
            "aarch64": "arm64",
            "amd64": "amd64",
            "x86_64": "amd64",
        }
        architectures_differ = normalized_arch.get(
            image_arch, image_arch
        ) != normalized_arch.get(engine_arch, engine_arch)
        if image_arch and engine_arch and architectures_differ and not self.platform:
            print(
                f"WARNING: Docker image architecture {image_arch} differs from "
                f"engine architecture {engine_arch}",
                flush=True,
            )

        commits_by_repo: dict[str, set[str]] = {}
        for problem in problems:
            repo_name = problem.repo.repo_name
            commits_by_repo.setdefault(repo_name, set()).update(
                test.commit_hash for test in problem.tests
            )

        checks = [
            "set -e",
            "command -v bash >/dev/null",
            "command -v git >/dev/null",
            "command -v jq >/dev/null",
            "command -v timeout >/dev/null",
            "command -v uv >/dev/null",
            "python --version",
        ]
        for repo_name, commits in sorted(commits_by_repo.items()):
            repo_dir = f"/workspace/{repo_name}"
            quoted_repo = shlex.quote(repo_dir)
            checks.append(f"test -d {quoted_repo}/.git")
            for commit in sorted(commits):
                quoted_commit = shlex.quote(commit)
                checks.append(
                    f"git -C {quoted_repo} cat-file -e {quoted_commit}^{{commit}}"
                )
                checks.append(f"git -C {quoted_repo} cat-file -e {quoted_commit}^")

        command = ["docker", "run", "--rm", *self._platform_args()]
        command.extend(
            ["--entrypoint", "/bin/bash", self.image, "-lc", "; ".join(checks)]
        )
        try:
            result = self._run(command, capture_output=True)
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "unknown validation error").strip()
            raise RuntimeError(
                f"Docker image {self.image!r} is not ready for this experiment: {details}"
            ) from exc

        version = result.stdout.strip().splitlines()
        version_text = version[-1] if version else "Python version unknown"
        print(
            f"Validated Docker image {self.image} "
            f"(image={image_arch or 'unknown'}, engine={engine_arch or 'unknown'}, "
            f"{version_text})",
            flush=True,
        )

    def _remove_stale_container(self, cluster: str) -> None:
        inspect = self._run(
            [
                "docker",
                "container",
                "inspect",
                cluster,
                "--format",
                '{{index .Config.Labels "gso.managed"}}',
            ],
            check=False,
            capture_output=True,
        )
        if inspect.returncode != 0:
            return
        if inspect.stdout.strip() != "true":
            raise RuntimeError(
                f"Container name {cluster!r} is already used by an unmanaged container"
            )
        self._run(["docker", "rm", "--force", cluster], capture_output=True)

    def launch_task(
        self,
        task_yaml: str,
        workspace: Path,
        cluster: str = "docker-gso",
        interactive: bool = False,
    ) -> None:
        with (Path(workspace) / task_yaml).open() as f:
            task = yaml.safe_load(f)
        run_command = task.get("run")
        if not isinstance(run_command, str) or not run_command.strip():
            raise ValueError(f"Task YAML has no run command: {task_yaml}")

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", cluster):
            raise ValueError(f"Invalid Docker container name: {cluster!r}")

        self._remove_stale_container(cluster)
        command = [
            "docker",
            "create",
            "--name",
            cluster,
            "--init",
            "--label",
            "gso.managed=true",
            "--label",
            f"gso.task={cluster}",
            "--workdir",
            "/workspace",
            *self._platform_args(),
        ]
        if self.cpus is not None:
            command.extend(["--cpus", str(self.cpus)])
        if self.memory:
            command.extend(["--memory", self.memory])
        for env_name in ("HF_TOKEN", "DEBUG_GSO"):
            if os.getenv(env_name):
                command.extend(["--env", env_name])

        # The image already contains /workspace/<repo>. Copy only the generated
        # phase scripts and tests, preserving the repository baked into it.
        command.extend([self.image, "/bin/bash", "-lc", f"set -e\n{run_command}"])
        self._run(command, capture_output=True)
        with self._lock:
            self._containers.add(cluster)

        try:
            self._run(
                [
                    "docker",
                    "cp",
                    f"{Path(workspace).resolve()}/.",
                    f"{cluster}:/workspace/",
                ],
                capture_output=True,
            )
            start_command = ["docker", "start"]
            if interactive:
                start_command.append("--attach")
            start_command.append(cluster)
            # Attached `docker start` returns the task's exit code. Collection
            # still needs to run for failed tasks, so inspect that code later.
            self._run(
                start_command,
                check=not interactive,
                capture_output=not interactive,
            )
        except Exception:
            if not self.keep_containers:
                self.cleanup_cluster(cluster)
            raise

        logger.info(
            f"Launched {task_yaml} in Docker container {cluster} from {workspace}"
        )

    def _inspect_state(self, cluster: str) -> dict:
        result = self._run(
            ["docker", "inspect", "--format", "{{json .State}}", cluster],
            capture_output=True,
        )
        return json.loads(result.stdout)

    def is_complete(self, workspace: Path, cluster: str = "docker-gso") -> bool:
        state = self._inspect_state(cluster)
        status = state.get("Status")
        if status in {"exited", "dead"}:
            return True
        if status in {"created", "running", "restarting", "paused"}:
            return False
        raise RuntimeError(f"Unexpected Docker status for {cluster}: {status!r}")

    def get_exit_code(self, cluster: str) -> int | None:
        state = self._inspect_state(cluster)
        if state.get("Status") not in {"exited", "dead"}:
            return None
        return int(state.get("ExitCode", 1))

    def get_results(
        self,
        workspace: Path,
        cluster: str = "docker-gso",
        *,
        expect_results: bool = True,
    ):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        logs = self._run(["docker", "logs", cluster], check=False, capture_output=True)
        log_path = self.artifact_dir / f"{cluster}.log"
        log_path.write_text((logs.stdout or "") + (logs.stderr or ""))

        if not expect_results:
            return (
                f"Container {cluster}: phase-1 validation; "
                f"result collection skipped (not expected); log={log_path}",
                [],
            )

        results_dir = Path(workspace) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        copied = self._run(
            ["docker", "cp", f"{cluster}:/workspace/results/.", str(results_dir)],
            check=False,
            capture_output=True,
        )
        results = collect_results(results_dir)
        message = (
            f"Container {cluster}: collected {len(results)} result(s); log={log_path}"
        )
        if copied.returncode != 0:
            detail = (copied.stderr or copied.stdout).strip()
            message += f"; result copy failed: {detail}"
        return message, results

    def cleanup_workspace(self, workspace: Path) -> None:
        shutil.rmtree(workspace, ignore_errors=True)
        logger.info(f"Deleted workspace: {workspace}")

    def cleanup_cluster(self, cluster: str, interactive: bool = False) -> None:
        if self.keep_containers:
            logger.info(f"Keeping Docker container: {cluster}")
            return
        self._run(
            ["docker", "rm", "--force", cluster],
            check=False,
            capture_output=True,
        )
        with self._lock:
            self._containers.discard(cluster)
        logger.info(f"Deleted Docker container: {cluster}")

    def cleanup_all_clusters(self, interactive: bool = False) -> None:
        with self._lock:
            containers = list(self._containers)
        for container in containers:
            self.cleanup_cluster(container, interactive=interactive)
