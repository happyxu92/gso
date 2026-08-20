import asyncio
from types import SimpleNamespace

from gso.collect.execute import execute
from gso.collect.execute.execute import (
    ExecutionManager,
    _completed_run_count,
    _reuse_existing_results,
)
from gso.data import Problem, Repo
from gso.utils.io import load_problems, save_problems


def make_problem(pid: str, api: str) -> Problem:
    return Problem(
        pid=pid,
        repo=Repo(
            repo_url="https://github.com/example/repo",
            repo_owner="example",
            repo_name="repo",
        ),
        api=api,
        py_version="3.12",
    )


def test_existing_successes_are_reused_and_failed_attempts_are_retried(tmp_path):
    previous = make_problem("repo-first", "first")
    previous.results = {
        "successful": [{"commit": "abcdef1", "test_file": "test_0.py"}],
        "failed": [],
    }
    results_path = tmp_path / "results.json"
    save_problems(results_path, [previous])

    current = make_problem("repo-first", "first")
    reused_runs = _reuse_existing_results([current], results_path)

    assert reused_runs == 1
    assert _completed_run_count(current) == 1
    assert current.results == {
        "successful": [{"commit": "abcdef1", "test_file": "test_0.py"}]
    }


def test_manager_only_creates_missing_problem_runs(tmp_path):
    completed = make_problem("repo-completed", "completed")
    completed.results = {"previous": [{"commit": "abcdef1"}]}
    pending = make_problem("repo-pending", "pending")
    created = []

    def create_workspace(problem, **kwargs):
        created.append((problem.pid, kwargs))
        workspace = tmp_path / problem.pid
        workspace.mkdir(exist_ok=True)
        return workspace

    runtime = SimpleNamespace(
        cluster_prefix="test",
        create_workspace=create_workspace,
        cleanup_workspace=lambda workspace: None,
    )
    manager = ExecutionManager(
        exp_id="resume",
        exp_dir=tmp_path,
        problems=[completed, pending],
        machines=1,
        runs=1,
        runtime=runtime,
        results_path=tmp_path / "results.json",
    )

    try:
        manager.initialize_problems()
        assert [pid for pid, _ in created] == ["repo-pending"]
        assert [state.problem.pid for state in manager.tasks.values()] == [
            "repo-pending"
        ]
    finally:
        manager.thread_pool.shutdown(wait=True, cancel_futures=True)


def test_async_main_uses_lowercase_default_docker_image(monkeypatch, tmp_path):
    exp_id = "MinerU"
    exp_dir = tmp_path / exp_id
    exp_dir.mkdir()
    problem = make_problem("MinerU-first", "first")
    save_problems(exp_dir / f"{exp_id}_problems.json", [problem])
    captured = {}

    class FakeDockerManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def validate(self, problems):
            assert problems == [problem]

    class FakeExecutionManager:
        def __init__(self, **kwargs):
            pass

        def initialize_problems(self):
            pass

        async def run(self):
            pass

    monkeypatch.setattr(execute, "EXPS_DIR", tmp_path)
    monkeypatch.setattr(execute, "DockerManager", FakeDockerManager)
    monkeypatch.setattr(execute, "ExecutionManager", FakeExecutionManager)
    monkeypatch.setattr(execute, "validate_problem_test_samples", lambda problems: None)

    asyncio.run(execute.async_main(exp_id, machines=1, runs=1, backend="docker"))

    assert captured["image"] == "gso-mineru:latest"
    assert (tmp_path / "MinerU").is_dir()


def test_async_main_returns_without_starting_backend_when_all_results_exist(
    monkeypatch, tmp_path
):
    exp_id = "resume"
    exp_dir = tmp_path / exp_id
    exp_dir.mkdir()
    problem = make_problem("repo-first", "first")
    save_problems(exp_dir / f"{exp_id}_problems.json", [problem])

    completed = problem.model_copy(deep=True)
    completed.results = {
        "failed": [],
        "previous": [{"commit": "abcdef1"}],
    }
    results_path = exp_dir / f"{exp_id}_results_docker.json"
    save_problems(results_path, [completed])

    monkeypatch.setattr(execute, "EXPS_DIR", tmp_path)

    class UnexpectedRuntime:
        def __init__(self, *args, **kwargs):
            raise AssertionError("backend should not be initialized")

    monkeypatch.setattr(execute, "DockerManager", UnexpectedRuntime)

    actual_path = asyncio.run(
        execute.async_main(exp_id, machines=1, runs=1, backend="docker")
    )

    assert actual_path == results_path
    saved = load_problems(results_path)
    assert saved[0].results == {"previous": [{"commit": "abcdef1"}]}
