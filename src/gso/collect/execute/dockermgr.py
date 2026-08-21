import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import yaml

from gso.collect.execute.helpers import collect_results
from gso.collect.execute.skymgr import SkyManager
from gso.harness.environment.patches import ensure_patch_dependencies
from gso.logger import logger


class DockerManager:
    """Run generated collection tasks in isolated local Docker containers."""

    cluster_prefix = "docker-gso"
    repository_dockerfile = """# syntax=docker/dockerfile:1
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# Install the compiler-cache client once in the repository image. The cache
# contents themselves live on the host and are mounted into task containers.
RUN if ! command -v ccache >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then \\
        apt-get update \\
        && apt-get install -y --no-install-recommends ccache \\
        && rm -rf /var/lib/apt/lists/*; \\
    fi
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
# Keep this layer identical to the remote-clone image so Docker can reuse it.
RUN if ! command -v ccache >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then \\
        apt-get update \\
        && apt-get install -y --no-install-recommends ccache \\
        && rm -rf /var/lib/apt/lists/*; \\
    fi
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
        cache_dir: Path | None = None,
        prepared_commit_hash: str | None = None,
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
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        )
        self.prepared_commit_hash = prepared_commit_hash
        self._containers: set[str] = set()
        self._lock = threading.Lock()

    def create_workspace(
        self, problem, phase1_only: bool = False, test_timeout: int = 300
    ) -> Path:
        # Prepared images are already checked out and installed at the candidate's
        # parent revision. The phase script verifies that revision instead of
        # mutating the checkout or rerunning installation commands.
        return SkyManager.create_workspace(
            problem,
            phase1_only=phase1_only,
            test_timeout=test_timeout,
            prepared_environment=self.prepared_commit_hash is not None,
        )

    def _platform_args(self) -> list[str]:
        return ["--platform", self.platform] if self.platform else []

    def _cache_mount_args(self) -> list[str]:
        if self.cache_dir is None:
            return []
        # Package and compiler caches are safe to share. Checkout-specific state
        # such as .venv, build/, dist/, and Rust target/ stays in the container.
        for child in ("ccache", "sccache", "pip", "uv"):
            (self.cache_dir / child).mkdir(parents=True, exist_ok=True)
        return [
            "--mount",
            f"type=bind,source={self.cache_dir},target=/gso-cache",
            "--env",
            "CCACHE_DIR=/gso-cache/ccache",
            "--env",
            "CCACHE_BASEDIR=/workspace",
            "--env",
            "CCACHE_COMPILERCHECK=content",
            "--env",
            "CCACHE_NOHASHDIR=true",
            "--env",
            "PIP_CACHE_DIR=/gso-cache/pip",
            "--env",
            "UV_CACHE_DIR=/gso-cache/uv",
            "--env",
            "SCCACHE_DIR=/gso-cache/sccache",
        ]

    @staticmethod
    def _cache_shell_setup() -> str:
        return r"""
if [ -f /gso-prepared-env.sh ]; then
    source /gso-prepared-env.sh
fi
if command -v ccache >/dev/null 2>&1; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export USE_CCACHE=1
    echo "GSO build cache: ccache (${CCACHE_DIR:-disabled})"
elif command -v sccache >/dev/null 2>&1; then
    export CMAKE_C_COMPILER_LAUNCHER=sccache
    export CMAKE_CXX_COMPILER_LAUNCHER=sccache
    export RUSTC_WRAPPER=sccache
    echo "GSO build cache: sccache (${SCCACHE_DIR:-disabled})"
else
    echo "GSO build cache: compiler cache unavailable; using package caches only"
fi
"""

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
                raise ValueError(
                    f"Docker repository path is not a directory: {local_repo}"
                )
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
            (
                self.local_repository_dockerfile
                if local_repo is not None
                else self.repository_dockerfile
            ),
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

    def _prepared_image_name(
        self,
        candidate_commit_hash: str,
        install_commands: list[str],
        session_id: str,
    ) -> str:
        image_without_digest = self.image.split("@", 1)[0]
        last_slash = image_without_digest.rfind("/")
        last_colon = image_without_digest.rfind(":")
        repository = (
            image_without_digest[:last_colon]
            if last_colon > last_slash
            else image_without_digest
        )
        safe_session = re.sub(r"[^a-z0-9]+", "-", session_id.lower()).strip("-")
        safe_session = (safe_session or "session")[:12]
        environment_key = json.dumps(
            {
                "source_image": self.image,
                "candidate_commit": candidate_commit_hash,
                "install_commands": install_commands,
                "platform": self.platform,
            },
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(environment_key).hexdigest()[:12]
        return (
            f"{repository}:prepared-{safe_session}-"
            f"{candidate_commit_hash[:12].lower()}-{digest}"
        )

    def prepare_commit_image(
        self,
        problem,
        candidate_commit_hash: str,
        *,
        session_id: str,
    ) -> str:
        """Checkout and install one candidate parent into a reusable image.

        Installation runs in a disposable container. Only its filesystem is
        committed; package/compiler caches remain host mounts, and credentials
        are supplied to ``docker exec`` rather than stored in container config.
        """
        if self.prepared_commit_hash is not None:
            raise ValueError(
                "Cannot prepare a commit image from another prepared image"
            )

        repo_name = problem.repo.repo_name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo_name):
            raise ValueError(f"Invalid repository name: {repo_name!r}")
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", candidate_commit_hash):
            raise ValueError(
                f"Invalid candidate commit hash: {candidate_commit_hash!r}"
            )

        install_commands = ensure_patch_dependencies(
            problem.pid, list(problem.install_commands)
        )
        prepared_image = self._prepared_image_name(
            candidate_commit_hash, install_commands, session_id
        )
        safe_session = re.sub(r"[^a-z0-9]+", "-", session_id.lower()).strip("-")
        container = (
            f"gso-prepare-{(safe_session or 'session')[:10]}-"
            f"{candidate_commit_hash[:12].lower()}-{uuid.uuid4().hex[:6]}"
        )
        repo_dir = f"/workspace/{repo_name}"
        quoted_repo_dir = shlex.quote(repo_dir)
        quoted_commit = shlex.quote(f"{candidate_commit_hash}^")
        quoted_candidate_object = shlex.quote(f"{candidate_commit_hash}^{{commit}}")
        quoted_venv = shlex.quote(f"{repo_dir}/.venv")
        install_script = "\n".join(install_commands)
        script = f"""set -euxo pipefail
{self._cache_shell_setup()}
cd {quoted_repo_dir}
git reset --hard
git clean -ffdx
git checkout --detach {quoted_commit}
git submodule update --init --recursive
{install_script}
git rev-parse {quoted_candidate_object} > /gso-prepared-candidate
if [ -x {quoted_venv}/bin/python ]; then
    cat > /gso-prepared-env.sh <<'GSO_PREPARED_ENV'
export VIRTUAL_ENV={repo_dir}/.venv
export PATH={repo_dir}/.venv/bin:$PATH
GSO_PREPARED_ENV
else
    rm -f /gso-prepared-env.sh
fi
"""

        create_command = [
            "docker",
            "create",
            "--name",
            container,
            "--init",
            "--label",
            "gso.managed=true",
            "--label",
            f"gso.prepared-session={safe_session or 'session'}",
            "--workdir",
            repo_dir,
            *self._platform_args(),
            *self._cache_mount_args(),
        ]
        if self.cpus is not None:
            create_command.extend(["--cpus", str(self.cpus)])
        if self.memory:
            create_command.extend(["--memory", self.memory])
        create_command.extend(
            [
                self.image,
                "/bin/bash",
                "-lc",
                "while :; do sleep 3600; done",
            ]
        )

        log_dir = self.artifact_dir / "prepared_images"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{candidate_commit_hash[:12]}.log"
        try:
            self._run(create_command, capture_output=True)
            self._run(["docker", "start", container], capture_output=True)
            exec_command = ["docker", "exec"]
            if os.getenv("HF_TOKEN"):
                exec_command.extend(["--env", "HF_TOKEN"])
            exec_command.extend([container, "/bin/bash", "-lc", script])
            installed = self._run(exec_command, check=False, capture_output=True)
            log_path.write_text(
                (installed.stdout or "") + (installed.stderr or ""), encoding="utf-8"
            )
            if installed.returncode != 0:
                detail = (
                    installed.stderr or installed.stdout or "unknown error"
                ).strip()
                raise RuntimeError(
                    f"Failed to prepare {candidate_commit_hash[:12]}: "
                    f"{detail[-4000:]}; log={log_path}"
                )

            committed = self._run(
                [
                    "docker",
                    "commit",
                    "--pause=true",
                    "--change",
                    "LABEL gso.prepared=true",
                    "--change",
                    f"LABEL gso.candidate-commit={candidate_commit_hash}",
                    container,
                    prepared_image,
                ],
                check=False,
                capture_output=True,
            )
            if committed.returncode != 0:
                detail = (
                    committed.stderr or committed.stdout or "unknown error"
                ).strip()
                raise RuntimeError(
                    f"Failed to commit prepared image {prepared_image}: {detail[-4000:]}"
                )
        finally:
            self._run(
                ["docker", "rm", "--force", container],
                check=False,
                capture_output=True,
            )

        print(
            f"Prepared Docker image {prepared_image} for "
            f"{candidate_commit_hash[:12]}^; log={log_path}",
            flush=True,
        )
        return prepared_image

    def remove_image(self) -> None:
        """Remove this runtime's image without touching shared cache directories."""
        result = self._run(
            ["docker", "image", "rm", "--force", self.image],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise RuntimeError(f"Failed to remove Docker image {self.image}: {detail}")

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

    def start_validation_container(self, cluster: str) -> None:
        """Start a long-lived container used for serial generation validation."""
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
        command.extend(self._cache_mount_args())
        if self.cpus is not None:
            command.extend(["--cpus", str(self.cpus)])
        if self.memory:
            command.extend(["--memory", self.memory])
        for env_name in ("HF_TOKEN", "DEBUG_GSO"):
            if os.getenv(env_name):
                command.extend(["--env", env_name])
        command.extend(
            [
                self.image,
                "/bin/bash",
                "-lc",
                "trap 'exit 0' TERM INT; while true; do sleep 3600; done",
            ]
        )
        self._run(command, capture_output=True)
        try:
            self._run(["docker", "start", cluster], capture_output=True)
        except Exception:
            self._run(
                ["docker", "rm", "--force", cluster],
                check=False,
                capture_output=True,
            )
            raise
        with self._lock:
            self._containers.add(cluster)

    def validation_container_running(self, cluster: str) -> bool:
        """Return whether a managed validation container is still running."""
        result = self._run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                cluster,
            ],
            check=False,
            capture_output=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def copy_validation_workspace(
        self, cluster: str, workspace: Path, destination: str
    ) -> None:
        """Copy one isolated test workspace into a validation container."""
        mkdir = self._run(
            ["docker", "exec", cluster, "mkdir", "-p", destination],
            check=False,
            capture_output=True,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(
                f"Failed to create validation workspace {destination}: "
                f"{(mkdir.stderr or mkdir.stdout).strip()}"
            )
        self._run(
            [
                "docker",
                "cp",
                f"{Path(workspace).resolve()}/.",
                f"{cluster}:{destination}/",
            ],
            capture_output=True,
        )

    def exec_validation_command(
        self, cluster: str, command: str
    ) -> subprocess.CompletedProcess:
        """Execute one validation command with the prepared environment loaded."""
        shell_command = f"{self._cache_shell_setup()}\n{command}"
        return self._run(
            ["docker", "exec", cluster, "/bin/bash", "-lc", shell_command],
            check=False,
            capture_output=True,
        )

    def close_validation_container(self, cluster: str) -> None:
        """Stop a persistent validation container and remove it unless retained."""
        if self.keep_containers:
            self._run(
                ["docker", "stop", "--time", "5", cluster],
                check=False,
                capture_output=True,
            )
            logger.info(f"Stopped and kept Docker validation container: {cluster}")
            return
        self.cleanup_cluster(cluster)

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
        command.extend(self._cache_mount_args())
        cache_setup = self._cache_shell_setup()
        if self.cpus is not None:
            command.extend(["--cpus", str(self.cpus)])
        if self.memory:
            command.extend(["--memory", self.memory])
        for env_name in ("HF_TOKEN", "DEBUG_GSO"):
            if os.getenv(env_name):
                command.extend(["--env", env_name])

        # The image already contains /workspace/<repo>. Copy only the generated
        # phase scripts and tests, preserving the repository baked into it.
        command.extend(
            [self.image, "/bin/bash", "-lc", f"set -e\n{cache_setup}{run_command}"]
        )
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
