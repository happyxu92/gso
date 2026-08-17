import argparse
import asyncio
import functools
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from gso.collect.execute.dockermgr import DockerManager
from gso.collect.execute.helpers import resolve_results_path
from gso.collect.execute.skymgr import SkyManager
from gso.collect.generate.helpers import validate_problem_test_samples
from gso.constants import EXPS_DIR
from gso.data import PerformanceCommit, Problem, Tests
from gso.utils.io import load_problems, save_problems


@dataclass
class TaskState:
    problem: Problem
    workspace: Path
    cluster: str
    run_index: int
    is_complete: bool = False
    results_collected: bool = False
    launching: bool = False
    cleaning: bool = False
    cleaned: bool = False
    failed: bool = False
    error: str | None = None
    status_errors: int = 0


@dataclass(frozen=True)
class GeneratedTestExecutionConfig:
    """Execution settings used while validating tests during generation."""

    exp_id: str
    backend: str = "docker"
    docker_image: str | None = None
    docker_cpus: float | None = None
    docker_memory: str | None = None
    docker_platform: str | None = None
    docker_base_image: str | None = None
    docker_repo_path: str | None = None
    rebuild_docker_image: bool = False
    keep_containers: bool = False
    keep_workspaces: bool = False
    poll_interval: float | None = None


@dataclass(frozen=True)
class GeneratedTestExecutionResult:
    """Result returned to the generator after executing one candidate test."""

    passed: bool
    error: str | None = None
    log_path: str | None = None


class ExecutionManager:
    def __init__(
        self,
        exp_id: str,
        exp_dir: Path,
        problems: list[Problem],
        machines: int,
        runs: int,
        runtime,
        results_path: Path,
        interactive: bool = False,
        poll_interval: float = 5,
        keep_workspaces: bool = False,
    ):
        if machines < 1:
            raise ValueError("machines must be at least 1")
        if runs < 1:
            raise ValueError("runs must be at least 1")
        if interactive and machines != 1:
            raise ValueError("interactive mode requires --machines 1")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than 0")

        self.exp_id = exp_id
        self.exp_dir = exp_dir
        self.problems = problems
        self.machines = machines
        self.runs = runs
        self.runtime = runtime
        self.results_path = results_path
        self.interactive = interactive
        self.poll_interval = poll_interval
        self.keep_workspaces = keep_workspaces

        self.cluster_counter = 0
        self.execution_id = uuid.uuid4().hex[:8]
        self.tasks: dict[str, TaskState] = {}
        self.active_clusters: set[str] = set()
        self.completed_clusters: set[str] = set()
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=max(2, machines * 2))
        self.last_progress_summary: str | None = None
        self.last_progress_print = 0.0

    def get_next_cluster_name(self) -> str:
        safe_exp_id = re.sub(r"[^a-z0-9-]+", "-", self.exp_id.lower()).strip("-")
        safe_exp_id = (safe_exp_id or "experiment")[:30]
        with self.lock:
            name = (
                f"{self.runtime.cluster_prefix}-{safe_exp_id}-"
                f"{self.execution_id}-{self.cluster_counter}"
            )
            self.cluster_counter += 1
            return name

    def initialize_problems(self) -> None:
        """Create one isolated workspace for every problem/run combination."""
        errors = []
        for problem in self.problems:
            for run_idx in range(self.runs):
                cluster = self.get_next_cluster_name()
                try:
                    workspace = Path(self.runtime.create_workspace(problem))
                except Exception as exc:
                    errors.append(f"{problem.pid} run {run_idx + 1}: {exc}")
                    continue

                self.tasks[cluster] = TaskState(
                    problem=problem,
                    workspace=workspace,
                    cluster=cluster,
                    run_index=run_idx,
                )

        if errors:
            for state in self.tasks.values():
                self.runtime.cleanup_workspace(state.workspace)
            self.tasks.clear()
            raise RuntimeError(
                "Failed to create task workspace(s):\n" + "\n".join(errors)
            )
        if not self.tasks:
            raise RuntimeError("No execution tasks were created")

    def _result_key(self, state: TaskState) -> str:
        return f"{state.cluster}_run{state.run_index}"

    def _mark_failed(self, state: TaskState, error: str) -> None:
        state.failed = True
        state.error = error
        state.is_complete = True
        if not state.results_collected:
            state.problem.add_results(key=self._result_key(state), results=[])
        state.results_collected = True
        with self.lock:
            self.active_clusters.discard(state.cluster)
            self.completed_clusters.add(state.cluster)
        print(f"FAILED {state.problem.pid}: {error}", flush=True)

    async def launch_task_async(self, cluster: str, state: TaskState) -> None:
        if state.launching or state.results_collected:
            return

        state.launching = True
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.thread_pool,
                functools.partial(
                    self.runtime.launch_task,
                    f"{state.problem.pid}_task.yaml",
                    state.workspace,
                    cluster=cluster,
                    interactive=self.interactive,
                ),
            )
            with self.lock:
                self.active_clusters.add(cluster)
            print(
                f"Launched {state.problem.pid} "
                f"(run {state.run_index + 1}/{self.runs}) as {cluster}",
                flush=True,
            )
        except Exception as exc:
            self._mark_failed(state, f"launch failed: {exc}")
        finally:
            state.launching = False

    async def cleanup_cluster_async(self, cluster: str, state: TaskState) -> None:
        if state.cleaning or state.cleaned:
            return

        state.cleaning = True
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.thread_pool,
                functools.partial(
                    self.runtime.cleanup_cluster, cluster, interactive=False
                ),
            )
            state.cleaned = True
        except Exception as exc:
            print(f"WARNING: failed to clean up {cluster}: {exc}", flush=True)
        finally:
            state.cleaning = False

    async def launch_available_tasks(self) -> None:
        with self.lock:
            available_slots = self.machines - len(self.active_clusters)
            pending_tasks = [
                (cluster, state)
                for cluster, state in self.tasks.items()
                if cluster not in self.active_clusters
                and cluster not in self.completed_clusters
                and not state.launching
                and not state.results_collected
            ][: max(0, available_slots)]

        if pending_tasks:
            await asyncio.gather(
                *(
                    self.launch_task_async(cluster, state)
                    for cluster, state in pending_tasks
                )
            )

    async def check_completion(self) -> None:
        cleanup_tasks = []
        completed = set()
        loop = asyncio.get_running_loop()

        for cluster in list(self.active_clusters):
            state = self.tasks[cluster]
            try:
                is_complete = await loop.run_in_executor(
                    self.thread_pool,
                    functools.partial(
                        self.runtime.is_complete, state.workspace, cluster
                    ),
                )
                state.status_errors = 0
            except Exception as exc:
                state.status_errors += 1
                print(
                    f"WARNING: status check failed for {cluster} "
                    f"({state.status_errors}/3): {exc}",
                    flush=True,
                )
                if state.status_errors >= 3:
                    self._mark_failed(state, f"status checks failed: {exc}")
                    cleanup_tasks.append(self.cleanup_cluster_async(cluster, state))
                continue

            if not is_complete:
                continue

            state.is_complete = True
            completed.add(cluster)
            try:
                message, results = await loop.run_in_executor(
                    self.thread_pool,
                    functools.partial(
                        self.runtime.get_results, state.workspace, cluster
                    ),
                )
                state.problem.add_results(key=self._result_key(state), results=results)
                state.results_collected = True

                exit_code = self.runtime.get_exit_code(cluster)
                if exit_code not in (None, 0):
                    state.failed = True
                    state.error = f"task exited with code {exit_code}"
                elif not results:
                    state.failed = True
                    state.error = "task produced no usable result files"

                status = "FAILED" if state.failed else "Completed"
                detail = f"; {state.error}" if state.error else ""
                print(
                    f"{status} {state.problem.pid} "
                    f"(run {state.run_index + 1}/{self.runs}): {message}{detail}",
                    flush=True,
                )
            except Exception as exc:
                self._mark_failed(state, f"result collection failed: {exc}")

            cleanup_tasks.append(self.cleanup_cluster_async(cluster, state))

        with self.lock:
            self.active_clusters -= completed
            self.completed_clusters |= completed

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)

    def save(self) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        save_problems(self.results_path, self.problems)

    def cleanup(self) -> None:
        # Clean only resources created by this invocation. In particular, do
        # not use `sky down -a` or remove unrelated Docker containers.
        for cluster, state in self.tasks.items():
            if not state.cleaned:
                try:
                    self.runtime.cleanup_cluster(cluster, interactive=False)
                    state.cleaned = True
                except Exception as exc:
                    print(f"WARNING: cleanup failed for {cluster}: {exc}", flush=True)

        if self.keep_workspaces:
            # Surface each retained workspace on stdout with enough pid/commit
            # context to map it back to a specific attempt. By default these per
            # test host workspaces (phase scripts + <quick_hash>/test_*.py) are
            # rmtree'd as soon as the container exits, leaving only the container
            # log. Operators set --keep-workspaces specifically to tell a test
            # that never ran from a path/glob mistake (e.g. the //<hash>/test_*.py
            # signature) by inspecting the retained layout, so announce where it
            # lives instead of leaving it as an anonymous tempdir to find.
            for state in self.tasks.values():
                quick_hashes = ",".join(
                    test.quick_hash for test in state.problem.tests
                )
                print(
                    f"Kept workspace for {state.problem.pid} "
                    f"(commit(s) {quick_hashes}, run {state.run_index + 1}): "
                    f"{state.workspace}",
                    flush=True,
                )
            return

        for state in self.tasks.values():
            self.runtime.cleanup_workspace(state.workspace)

    def all_tasks_complete(self) -> bool:
        return bool(self.tasks) and all(
            state.results_collected for state in self.tasks.values()
        )

    def get_progress_summary(self) -> str:
        total_tasks = len(self.tasks)
        completed_tasks = len(self.completed_clusters)
        active_tasks = len(self.active_clusters)
        failed_tasks = sum(state.failed for state in self.tasks.values())
        pending_tasks = total_tasks - completed_tasks - active_tasks
        return (
            f"Progress: {completed_tasks}/{total_tasks} completed | "
            f"Active: {active_tasks} | Pending: {pending_tasks} | Failed: {failed_tasks}"
        )

    def print_progress(self) -> None:
        summary = self.get_progress_summary()
        now = time.monotonic()
        if (
            summary != self.last_progress_summary
            or now - self.last_progress_print >= 30
        ):
            print(summary, flush=True)
            self.last_progress_summary = summary
            self.last_progress_print = now

    async def run(self) -> None:
        saved_count = 0
        try:
            while not self.all_tasks_complete():
                await self.launch_available_tasks()
                await self.check_completion()
                self.print_progress()

                if len(self.completed_clusters) - saved_count >= 10:
                    self.save()
                    saved_count = len(self.completed_clusters)

                if not self.all_tasks_complete():
                    await asyncio.sleep(self.poll_interval)
        finally:
            try:
                self.save()
            finally:
                try:
                    self.cleanup()
                finally:
                    self.thread_pool.shutdown(wait=True, cancel_futures=True)

        failures = [state for state in self.tasks.values() if state.failed]
        if failures:
            details = "; ".join(
                f"{state.problem.pid}: {state.error or 'unknown failure'}"
                for state in failures
            )
            raise RuntimeError(f"{len(failures)} task(s) failed: {details}")


def _create_runtime(config: GeneratedTestExecutionConfig, exp_dir: Path):
    """Create the execution backend shared by batch and generation validation."""
    if config.backend == "docker":
        return DockerManager(
            image=config.docker_image or f"gso-{config.exp_id}:latest",
            artifact_dir=exp_dir / "docker_logs" / "generation_validation",
            cpus=config.docker_cpus,
            memory=config.docker_memory,
            platform=config.docker_platform,
            keep_containers=config.keep_containers,
            repository_base_image=config.docker_base_image,
            repository_path=(
                Path(config.docker_repo_path).expanduser()
                if config.docker_repo_path
                else None
            ),
            rebuild_repository_image=config.rebuild_docker_image,
        )
    if config.backend == "sky":
        return SkyManager()
    raise ValueError("backend must be 'sky' or 'docker'")


def prepare_generated_test_execution(
    problems: list[Problem], config: GeneratedTestExecutionConfig
) -> GeneratedTestExecutionConfig:
    """Prepare the backend once, then return settings safe for per-test workers."""
    exp_dir = EXPS_DIR / config.exp_id
    runtime = _create_runtime(config, exp_dir)

    # Docker validation discovers candidate revisions through Problem.tests. At
    # this point generation has not produced tests yet, so create commit-only
    # placeholders solely for the image/commit preflight.
    validation_problems = []
    for problem in problems:
        validation_problem = problem.model_copy(deep=True)
        validation_problem.set_tests(
            [Tests.from_commit(commit) for commit in validation_problem.commits]
        )
        validation_problems.append(validation_problem)
    runtime.validate(validation_problems)

    # The repository image is now ready. Per-test workers must reuse it rather
    # than racing to rebuild the same tag for every generated sample.
    if config.backend == "docker" and config.docker_base_image is not None:
        return GeneratedTestExecutionConfig(
            **{
                **config.__dict__,
                "docker_base_image": None,
                "docker_repo_path": None,
                "rebuild_docker_image": False,
            }
        )
    return config


def _execution_error_detail(
    manager: ExecutionManager, runtime, manager_error: Exception | None
) -> tuple[str, str | None]:
    """Collect concise backend diagnostics suitable for an LLM retry prompt."""
    details = []
    if manager_error is not None:
        details.append(str(manager_error))
    for state in manager.tasks.values():
        if state.error:
            details.append(state.error)

    log_path = None
    artifact_dir = getattr(runtime, "artifact_dir", None)
    if artifact_dir is not None:
        for state in manager.tasks.values():
            candidate = Path(artifact_dir) / f"{state.cluster}.log"
            if not candidate.exists():
                continue
            log_path = str(candidate)
            log_text = candidate.read_text(encoding="utf-8", errors="replace")
            if log_text.strip():
                details.append(f"Execution log tail:\n{log_text[-12000:]}")

    if not details:
        details.append(
            "The test did not produce a successful candidate-commit result. "
            "It may have failed setup, reference execution, or equivalence checking."
        )
    return "\n\n".join(details), log_path


async def _async_evaluate_generated_test(
    problem: Problem,
    commit: PerformanceCommit,
    generated_test: str,
    config: GeneratedTestExecutionConfig,
) -> GeneratedTestExecutionResult:
    """Execute one generated test against a commit parent and candidate."""
    try:
        validation_problem = problem.model_copy(deep=True)
        validation_problem.commits = [commit]
        commit_tests = Tests.from_commit(commit)
        commit_tests.add_sample(generated_test)
        validation_problem.set_tests([commit_tests])
        validation_problem.clear_results()
        validate_problem_test_samples([validation_problem])

        exp_dir = EXPS_DIR / config.exp_id
        runtime = _create_runtime(config, exp_dir)
        runtime.validate([validation_problem])
    except Exception as exc:
        return GeneratedTestExecutionResult(
            passed=False,
            error=(
                "Execution backend validation failed: " f"{type(exc).__name__}: {exc}"
            ),
        )
    effective_poll_interval = (
        (1 if config.backend == "docker" else 5)
        if config.poll_interval is None
        else config.poll_interval
    )

    # This file is diagnostic-only and remains separate from formal batch results.
    # A unique name also prevents parallel commit workers from overwriting it.
    results_path = (
        exp_dir
        / "generation_validation"
        / f"{problem.pid}_{commit.quick_hash()}_{uuid.uuid4().hex[:8]}.json"
    )
    manager = ExecutionManager(
        exp_id=config.exp_id,
        exp_dir=exp_dir,
        problems=[validation_problem],
        machines=1,
        runs=1,
        runtime=runtime,
        results_path=results_path,
        poll_interval=effective_poll_interval,
        keep_workspaces=config.keep_workspaces,
    )
    manager_error = None
    try:
        manager.initialize_problems()
        await manager.run()
    except Exception as exc:
        manager_error = exc
        if not manager.tasks:
            try:
                manager.thread_pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass

    collected_results = [
        result
        for run_results in validation_problem.results.values()
        for result in run_results
    ]
    passed = any(
        result.get("commit") in {commit.commit_hash, commit.quick_hash()}
        and "base_result" in result
        and "commit_result" in result
        and result.get("base_result") is not None
        and result.get("commit_result") is not None
        for result in collected_results
    )
    if passed and manager_error is None:
        return GeneratedTestExecutionResult(passed=True)

    error, log_path = _execution_error_detail(manager, runtime, manager_error)
    return GeneratedTestExecutionResult(passed=False, error=error, log_path=log_path)


def evaluate_generated_test(
    problem: Problem,
    commit: PerformanceCommit,
    generated_test: str,
    config: GeneratedTestExecutionConfig,
) -> GeneratedTestExecutionResult:
    """Synchronous entry point used by generation workers."""
    return asyncio.run(
        _async_evaluate_generated_test(problem, commit, generated_test, config)
    )


async def async_main(
    exp_id: str,
    machines: int,
    runs: int,
    specific_api: str | None = None,
    exp_yaml: str | None = None,
    interactive: bool = False,
    backend: str = "sky",
    docker_image: str | None = None,
    docker_cpus: float | None = None,
    docker_memory: str | None = None,
    docker_platform: str | None = None,
    docker_base_image: str | None = None,
    docker_repo_path: str | None = None,
    rebuild_docker_image: bool = False,
    keep_containers: bool = False,
    keep_workspaces: bool = False,
    poll_interval: float | None = None,
    results_file: str | None = None,
) -> Path:
    if backend not in {"sky", "docker"}:
        raise ValueError("backend must be 'sky' or 'docker'")

    exp_dir = EXPS_DIR / exp_id
    problems_path = exp_dir / f"{exp_id}_problems.json"
    all_problems = load_problems(problems_path)

    if exp_yaml is not None:
        exp_yaml_path = Path(exp_yaml).expanduser()
        if not exp_yaml_path.exists():
            raise ValueError(f"Experiment YAML not found: {exp_yaml_path}")
        with exp_yaml_path.open() as f:
            exp_data = yaml.safe_load(f) or {}
        for problem in all_problems:
            problem.target_commit = exp_data.get("target_commit", "main")
            problem.install_commands = list(exp_data.get("install_commands", []))

    if specific_api:
        problems = [problem for problem in all_problems if problem.api == specific_api]
        if not problems:
            raise ValueError(f"No problem found for API: {specific_api}")
    else:
        problems = all_problems
    if not problems:
        raise ValueError(f"No problems found in {problems_path}")

    validate_problem_test_samples(problems)

    results_path = resolve_results_path(
        exp_dir, exp_id, backend=backend, results_file=results_file
    )
    if backend == "docker":
        image = docker_image or f"gso-{exp_id}:latest"
        runtime = DockerManager(
            image=image,
            artifact_dir=exp_dir / "docker_logs",
            cpus=docker_cpus,
            memory=docker_memory,
            platform=docker_platform,
            keep_containers=keep_containers,
            repository_base_image=docker_base_image,
            repository_path=(
                Path(docker_repo_path).expanduser() if docker_repo_path else None
            ),
            rebuild_repository_image=rebuild_docker_image,
        )
        effective_poll_interval = 1 if poll_interval is None else poll_interval
        if machines > 1:
            print(
                "WARNING: concurrent local containers can distort performance results; "
                "use --machines 1 for stable measurements",
                flush=True,
            )
    else:
        runtime = SkyManager()
        effective_poll_interval = 5 if poll_interval is None else poll_interval

    runtime.validate(problems)
    print(
        f"Executing {len(problems)} problem(s) with backend={backend}; "
        f"results={results_path}",
        flush=True,
    )

    manager = ExecutionManager(
        exp_id=exp_id,
        exp_dir=exp_dir,
        problems=problems,
        machines=machines,
        runs=runs,
        runtime=runtime,
        results_path=results_path,
        interactive=interactive,
        poll_interval=effective_poll_interval,
        keep_workspaces=keep_workspaces,
    )
    manager.initialize_problems()
    await manager.run()
    return results_path


def main(
    exp_id: str,
    machines: int,
    runs: int,
    specific_api: str | None = None,
    exp_yaml: str | None = None,
    interactive: bool = False,
    backend: str = "sky",
    docker_image: str | None = None,
    docker_cpus: float | None = None,
    docker_memory: str | None = None,
    docker_platform: str | None = None,
    docker_base_image: str | None = None,
    docker_repo_path: str | None = None,
    rebuild_docker_image: bool = False,
    keep_containers: bool = False,
    keep_workspaces: bool = False,
    poll_interval: float | None = None,
    results_file: str | None = None,
) -> Path:
    return asyncio.run(
        async_main(
            exp_id,
            machines,
            runs,
            specific_api,
            exp_yaml,
            interactive,
            backend,
            docker_image,
            docker_cpus,
            docker_memory,
            docker_platform,
            docker_base_image,
            docker_repo_path,
            rebuild_docker_image,
            keep_containers,
            keep_workspaces,
            poll_interval,
            results_file,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute generated performance tests with SkyPilot or local Docker"
    )
    parser.add_argument("-e", "--exp_id", required=True, help="Experiment ID")
    parser.add_argument("-a", "--api", default=None, help="Run one specific API")
    parser.add_argument(
        "-yp", "--exp_yaml", default=None, help="Optional experiment YAML override"
    )
    parser.add_argument(
        "--backend", choices=["sky", "docker"], default="sky", help="Execution backend"
    )
    parser.add_argument(
        "-m",
        "--machines",
        type=int,
        default=None,
        help="Max concurrent tasks (default: 2 for SkyPilot, 1 for Docker)",
    )
    parser.add_argument("-r", "--runs", type=int, default=1, help="Runs per problem")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument(
        "--results-file",
        default=None,
        help="Results filename relative to the experiment directory, or absolute path",
    )

    docker_group = parser.add_argument_group("local Docker")
    docker_group.add_argument(
        "--docker-image",
        default=None,
        help="Docker image (default: gso-<exp_id>:latest)",
    )
    docker_group.add_argument(
        "--docker-cpus", type=float, default=None, help="CPU limit per container"
    )
    docker_group.add_argument(
        "--docker-memory", default=None, help="Memory limit per container, e.g. 6g"
    )
    docker_group.add_argument(
        "--docker-platform", default=None, help="Docker platform, e.g. linux/amd64"
    )
    docker_group.add_argument(
        "--docker-base-image",
        default=None,
        help=(
            "Local base image used to build --docker-image by cloning the "
            "experiment repository"
        ),
    )
    docker_group.add_argument(
        "--docker-repo-path",
        default=None,
        help=(
            "Local Git checkout copied into the repository image; requires "
            "--docker-base-image"
        ),
    )
    docker_group.add_argument(
        "--rebuild-docker-image",
        action="store_true",
        help="Build the repository image without Docker layer cache",
    )
    docker_group.add_argument(
        "--keep-containers", action="store_true", help="Keep containers after execution"
    )
    docker_group.add_argument(
        "--keep-workspaces", action="store_true", help="Keep generated host workspaces"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Status polling interval in seconds",
    )
    args = parser.parse_args()

    machines = args.machines
    if machines is None:
        machines = 1 if args.backend == "docker" else 2

    main(
        args.exp_id,
        machines,
        args.runs,
        args.api,
        args.exp_yaml,
        args.interactive,
        args.backend,
        args.docker_image,
        args.docker_cpus,
        args.docker_memory,
        args.docker_platform,
        args.docker_base_image,
        args.docker_repo_path,
        args.rebuild_docker_image,
        args.keep_containers,
        args.keep_workspaces,
        args.poll_interval,
        args.results_file,
    )
