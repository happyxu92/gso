from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gso.collect.execute import execute
from gso.collect.execute.execute import GeneratedTestExecutionConfig
from gso.data import PerformanceCommit, Problem, Repo


def test_generation_validation_accepts_clean_phase1_exit(monkeypatch, tmp_path):
    commit = PerformanceCommit(
        commit_hash="abcdef1234567890",
        subject="speed up",
        message="speed up",
        date=datetime.now(timezone.utc),
    )
    problem = Problem(
        pid="repo-api",
        repo=Repo(
            repo_url="https://github.com/example/repo",
            repo_owner="example",
            repo_name="repo",
        ),
        api="api",
        py_version="3.12",
        commits=[commit],
    )

    runtime = SimpleNamespace(validate=lambda problems: None)
    monkeypatch.setattr(execute, "EXPS_DIR", tmp_path)
    monkeypatch.setattr(execute, "_create_runtime", lambda config, exp_dir: runtime)

    class FakeManager:
        def __init__(self, **kwargs):
            self.problems = kwargs["problems"]
            self.tasks = {"run0": SimpleNamespace(failed=False)}

        def initialize_problems(self):
            pass

        async def run(self):
            pass

    monkeypatch.setattr(execute, "ExecutionManager", FakeManager)

    monkeypatch.setattr(execute, "validate_problem_test_samples", lambda problems: None)

    result = execute.evaluate_generated_test(
        problem,
        commit,
        "def setup(): pass",
        GeneratedTestExecutionConfig(exp_id="unit"),
    )

    assert result.passed is True
