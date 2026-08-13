import json
from types import SimpleNamespace

import pytest
from r2e.llms.language_model import LanguageModel, LanguageModelStyle
from r2e.llms.llm_args import LLMArgs

from gso.collect.generate.args import PerfExpGenArgs
from gso.collect.generate.generate import (
    CommitGenerationTask,
    create_generation_problem,
    generate_commit_tests,
)
from gso.collect.generate.helpers import (
    GeneratedScenarioError,
    get_generated_scenarios,
)
from gso.data import PerformanceCommit, Problem, Repo, Tests as CommitTests
from gso.utils import llm as llm_utils


SCENARIO_TEMPLATE = {
    "title": "scenario",
    "workload": "realistic workload",
    "input_characteristics": "diverse inputs at realistic scale",
    "api_usage": "call the target API",
    "optimization_focus": "exercise the optimized path",
    "equivalence_strategy": "store and compare the returned value",
    "distinguishing_factor": "use a distinct workload shape",
}

VALID_TEST = """import timeit


def setup():
    return None


def experiment():
    return 1


def store_result(result, path):
    return None


def load_result(path):
    return 1


def check_equivalence(reference, current):
    assert reference == current


def run_test(eqcheck=False, reference=False, prefix=''):
    return 0.0
"""


def test_generation_defaults_to_four_commit_workers_and_five_tests():
    args = PerfExpGenArgs(yaml_path="unused.yaml")

    assert args.multiprocess == 4
    assert args.n == 5


def test_problem_uses_configured_install_commands():
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    configured_commands = ["uv pip install -e '.[test]'"]

    problem = create_generation_problem(
        repo,
        "target",
        [],
        {"py_version": "3.12", "install_commands": configured_commands},
    )

    assert problem.install_commands == configured_commands


def test_problem_uses_default_install_commands_when_not_configured():
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )

    problem = create_generation_problem(repo, "target", [], {"py_version": "3.12"})

    assert problem.install_commands == [
        "uv venv --python 3.12",
        "source .venv/bin/activate",
        "which python",
        "python --version",
        "uv pip install -e .",
        "uv pip install requests",
        "uv pip show repo",
    ]


def test_generated_scenarios_require_expected_count_and_fields():
    scenarios = [
        {**SCENARIO_TEMPLATE, "title": "first"},
        {**SCENARIO_TEMPLATE, "title": "second"},
    ]
    output = f"```json\n{json.dumps({'scenarios': scenarios})}\n```"

    parsed = get_generated_scenarios(output, expected_count=2)

    assert [scenario.title for scenario in parsed] == ["first", "second"]

    with pytest.raises(GeneratedScenarioError, match="expected 3 scenario"):
        get_generated_scenarios(output, expected_count=3)

    del scenarios[0]["workload"]
    with pytest.raises(GeneratedScenarioError, match="invalid scenario structure"):
        get_generated_scenarios(json.dumps({"scenarios": scenarios}), expected_count=2)


def test_streaming_runner_uses_one_choice_even_when_test_count_is_larger():
    args = LLMArgs(model_name="custom-model", n=5, use_cache=False)
    model = LanguageModel(
        model_name=args.model_name,
        style=LanguageModelStyle.OpenAI,
    )

    runner = llm_utils.StreamingOpenAIRunner(args, model)

    assert runner.client_kwargs["n"] == 1


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_streaming_runner_rejects_choices_without_content(monkeypatch, content):
    args = LLMArgs(model_name="custom-model", n=1, use_cache=False)
    model = LanguageModel(
        model_name=args.model_name,
        style=LanguageModelStyle.OpenAI,
    )
    runner = llm_utils.StreamingOpenAIRunner(args, model)

    class FakeStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(content=content),
                    )
                ]
            )

        def close(self):
            self.closed = True

    stream = FakeStream()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: stream)
        )
    )
    monkeypatch.setattr(runner, "_get_client", lambda: client)

    with pytest.raises(
        llm_utils.IncompleteStreamingResponse,
        match=r"without content.*\[0\]",
    ):
        runner._consume_stream([{"role": "user", "content": "prompt"}])

    assert stream.closed


def test_single_completion_disables_payload_level_parallelism(monkeypatch):
    args = LLMArgs(model_name="custom-model", n=5, multiprocess=30)
    observed = {}

    def fake_completions(request_args, payloads):
        observed["n"] = request_args.n
        observed["multiprocess"] = request_args.multiprocess
        observed["payloads"] = payloads
        return [["one choice"]]

    monkeypatch.setattr(llm_utils, "get_streaming_llm_completions", fake_completions)

    output = llm_utils.get_streaming_llm_completion(
        args, [{"role": "user", "content": "prompt"}]
    )

    assert output == "one choice"
    assert observed == {
        "n": 1,
        "multiprocess": 1,
        "payloads": [[{"role": "user", "content": "prompt"}]],
    }


def test_commit_worker_generates_scenarios_then_tests_sequentially(monkeypatch):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commit = PerformanceCommit(
        commit_hash="abcdef1234567890",
        subject="Optimize target API",
        message="Optimize target API",
        date="2024-01-01T00:00:00",
    )
    problem = Problem(pid="repo.target", repo=repo, api="target", commits=[commit])
    args = PerfExpGenArgs(
        yaml_path="unused.yaml",
        model_name="custom-model",
        n=2,
        multiprocess=30,
        use_cache=False,
    )

    def fake_prepare(task_args):
        _, _, prepared_commit, _ = task_args
        commit_tests = CommitTests.from_commit(prepared_commit)
        commit_tests.init_chat("test system", "commit context", "write test")
        return commit_tests

    scenario_output = json.dumps(
        {
            "scenarios": [
                {**SCENARIO_TEMPLATE, "title": "first"},
                {**SCENARIO_TEMPLATE, "title": "second"},
            ]
        }
    )
    responses = iter([scenario_output, VALID_TEST, VALID_TEST])
    payloads = []

    def fake_completion(request_args, payload):
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_streaming_llm_completion",
        fake_completion,
    )

    result = generate_commit_tests(
        CommitGenerationTask(
            problem_index=0,
            commit_index=0,
            repo=repo,
            problem=problem,
            commit=commit,
            args=args,
        )
    )

    assert result.error is None
    assert result.commit_tests is not None
    assert result.commit_tests.num_samples() == 2
    assert len(payloads) == 3
    assert payloads[0][0]["content"].startswith(
        "You are a performance-test scenario planner"
    )
    assert '"title":"first"' in payloads[1][-1]["content"].replace(" ", "")
    assert '"title":"second"' in payloads[2][-1]["content"].replace(" ", "")
    # Duplicate test code is intentionally accepted; no deduplication is performed.
    assert result.test_outputs == [VALID_TEST, VALID_TEST]


def test_commit_worker_retries_invalid_scenario_without_cache(monkeypatch):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commit = PerformanceCommit(
        commit_hash="abcdef1234567890",
        subject="Optimize target API",
        message="Optimize target API",
        date="2024-01-01T00:00:00",
    )
    problem = Problem(pid="repo.target", repo=repo, api="target", commits=[commit])
    args = PerfExpGenArgs(
        yaml_path="unused.yaml",
        model_name="custom-model",
        n=1,
        use_cache=True,
    )

    def fake_prepare(task_args):
        _, _, prepared_commit, _ = task_args
        commit_tests = CommitTests.from_commit(prepared_commit)
        commit_tests.init_chat("test system", "commit context", "write test")
        return commit_tests

    invalid_scenario = '{"scenarios": [{"title": "broken", "workload": unquoted}]}'
    valid_scenario = json.dumps({"scenarios": [SCENARIO_TEMPLATE]})
    responses = iter([invalid_scenario, valid_scenario, VALID_TEST])
    calls = []

    def fake_completion(request_args, payload):
        calls.append((request_args, payload))
        return next(responses)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_streaming_llm_completion",
        fake_completion,
    )

    result = generate_commit_tests(
        CommitGenerationTask(
            problem_index=0,
            commit_index=0,
            repo=repo,
            problem=problem,
            commit=commit,
            args=args,
        )
    )

    assert result.error is None
    assert result.commit_tests is not None
    assert result.commit_tests.num_samples() == 1
    assert len(calls) == 3
    assert calls[0][0].use_cache is True
    assert calls[1][0].use_cache is False
    retry_instruction = calls[1][1][-1]["content"]
    assert "semantic retry 1 of 3" in retry_instruction
    assert "invalid JSON" in retry_instruction


def test_commit_worker_retries_invalid_test_without_cache(monkeypatch):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commit = PerformanceCommit(
        commit_hash="abcdef1234567890",
        subject="Optimize target API",
        message="Optimize target API",
        date="2024-01-01T00:00:00",
    )
    problem = Problem(pid="repo.target", repo=repo, api="target", commits=[commit])
    args = PerfExpGenArgs(
        yaml_path="unused.yaml",
        model_name="custom-model",
        n=1,
        use_cache=True,
    )

    def fake_prepare(task_args):
        _, _, prepared_commit, _ = task_args
        commit_tests = CommitTests.from_commit(prepared_commit)
        commit_tests.init_chat("test system", "commit context", "write test")
        return commit_tests

    scenario_output = json.dumps({"scenarios": [SCENARIO_TEMPLATE]})
    invalid_test = "```python\nimport timeit\n```"
    responses = iter([scenario_output, invalid_test, VALID_TEST])
    calls = []

    def fake_completion(request_args, payload):
        calls.append((request_args, payload))
        return next(responses)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_streaming_llm_completion",
        fake_completion,
    )

    result = generate_commit_tests(
        CommitGenerationTask(
            problem_index=0,
            commit_index=0,
            repo=repo,
            problem=problem,
            commit=commit,
            args=args,
        )
    )

    assert result.error is None
    assert result.commit_tests is not None
    assert result.commit_tests.num_samples() == 1
    assert result.test_outputs == [invalid_test, VALID_TEST]
    assert calls[1][0].use_cache is True
    assert calls[2][0].use_cache is False
    retry_instruction = calls[2][1][-1]["content"]
    assert "semantic retry 1 of 3" in retry_instruction
    assert "missing required function" in retry_instruction
