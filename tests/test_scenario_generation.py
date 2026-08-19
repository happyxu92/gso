import json
import re
from types import SimpleNamespace

import pytest
import yaml
from r2e.llms.language_model import LanguageModel, LanguageModelStyle
from r2e.llms.llm_args import LLMArgs

from gso.collect.generate.args import PerfExpGenArgs
from gso.collect.generate.generate import (
    API_PREFLIGHT_RESULT_PREFIX,
    CommitGenerationResult,
    CommitGenerationTask,
    GENERATION_EVENT_PREFIX,
    GENERATION_PROGRESS_PREFIX,
    PerfExpGenerator,
    SEMANTIC_RETRY_COUNT,
    build_api_preflight_test,
    create_generation_problem,
    generate_commit_tests,
    load_existing_problems,
    preflight_generation_problems,
    request_install_commands_from_codex,
    save_problems_atomically,
    update_yaml_install_commands,
)
from gso.collect.execute.execute import (
    GeneratedTestExecutionConfig,
    GeneratedTestExecutionResult,
)
from gso.collect.generate.helpers import (
    GeneratedScenarioError,
    get_generated_scenario_and_test,
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


def combined_output(scenario=SCENARIO_TEMPLATE, test=VALID_TEST):
    return f"```json\n{json.dumps(scenario)}\n```\n\n```python\n{test}\n```"


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


def test_api_preflight_test_resolves_qualified_and_exported_apis(capsys):
    namespace = {}
    source = build_api_preflight_test(["json.dumps", "dumps"], "json")

    exec(compile(source, "<api-preflight>", "exec"), namespace)
    results = namespace["experiment"]()

    assert results["json.dumps"]["ok"] is True
    assert results["dumps"]["ok"] is True
    assert API_PREFLIGHT_RESULT_PREFIX in capsys.readouterr().out


def test_preflight_skips_only_unresolved_api_in_shared_commit_group(monkeypatch):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commit = PerformanceCommit(
        commit_hash="1111111234567890",
        subject="Optimize APIs",
        message="Optimize APIs",
        date="2024-01-01T00:00:00",
    )
    problems = [
        Problem(pid="repo-good", repo=repo, api="good", commits=[commit]),
        Problem(pid="repo-bad", repo=repo, api="bad", commits=[commit]),
    ]
    report = {
        "good": {"ok": True, "resolved": "repo.good"},
        "bad": {"ok": False, "error": "ImportError: missing API"},
    }
    calls = []

    def fake_evaluate(problem, checked_commit, test, config):
        calls.append((problem, checked_commit, test, config))
        return GeneratedTestExecutionResult(
            passed=False,
            error=(
                "execution failed\n"
                + API_PREFLIGHT_RESULT_PREFIX
                + json.dumps(report, separators=(",", ":"))
                + "\ntraceback"
            ),
        )

    monkeypatch.setattr(
        "gso.collect.generate.generate.evaluate_generated_test", fake_evaluate
    )
    config = GeneratedTestExecutionConfig(exp_id="repo")

    accepted, diagnostics = preflight_generation_problems(problems, config)

    assert [problem.api for problem in accepted] == ["good"]
    assert len(calls) == 1
    assert "good" in calls[0][2]
    assert "bad" in calls[0][2]
    assert diagnostics == [
        {
            "pid": "repo-bad",
            "api": "bad",
            "commit_hash": commit.commit_hash,
            "stage": "install_api_preflight",
            "error": "ImportError: missing API",
        }
    ]


def test_preflight_skips_commit_group_when_install_fails(monkeypatch):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commit = PerformanceCommit(
        commit_hash="1111111234567890",
        subject="Optimize API",
        message="Optimize API",
        date="2024-01-01T00:00:00",
    )
    problem = Problem(pid="repo-target", repo=repo, api="target", commits=[commit])
    monkeypatch.setattr(
        "gso.collect.generate.generate.evaluate_generated_test",
        lambda *args: GeneratedTestExecutionResult(
            passed=False, error="install command exited with status 1"
        ),
    )

    accepted, diagnostics = preflight_generation_problems(
        [problem], GeneratedTestExecutionConfig(exp_id="repo")
    )

    assert accepted == []
    assert diagnostics[0]["api"] == "target"
    assert diagnostics[0]["error"] == "install command exited with status 1"


def test_preflight_rejects_api_when_a_later_candidate_commit_fails(monkeypatch):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commits = [
        PerformanceCommit(
            commit_hash="1111111234567890",
            subject="First optimization",
            message="First optimization",
            date="2024-01-01T00:00:00",
        ),
        PerformanceCommit(
            commit_hash="2222222234567890",
            subject="Second optimization",
            message="Second optimization",
            date="2024-01-02T00:00:00",
        ),
    ]
    problem = Problem(pid="repo-target", repo=repo, api="target", commits=commits)
    checked_hashes = []

    def fake_evaluate(_problem, commit, _test, _config):
        checked_hashes.append(commit.commit_hash)
        if commit == commits[0]:
            return GeneratedTestExecutionResult(passed=True)
        return GeneratedTestExecutionResult(
            passed=False, error="API is unavailable on this parent commit"
        )

    monkeypatch.setattr(
        "gso.collect.generate.generate.evaluate_generated_test", fake_evaluate
    )

    accepted, diagnostics = preflight_generation_problems(
        [problem], GeneratedTestExecutionConfig(exp_id="repo")
    )

    assert accepted == []
    assert checked_hashes == [commit.commit_hash for commit in commits]
    assert diagnostics[0]["commit_hash"] == commits[1].commit_hash


def test_codex_install_command_inference_uses_local_repo_and_schema(
    tmp_path, monkeypatch
):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    expected_commands = [
        "uv venv --python 3.12",
        "source .venv/bin/activate",
        "uv pip install -e '.[test]' requests",
    ]

    def fake_run(command, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        with open(output_path, "w") as file:
            json.dump({"install_commands": expected_commands}, file)
        assert command[command.index("--cd") + 1] == str(repo_path.resolve())
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert "Python 3.12 is requested" in command[-1]
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("gso.collect.generate.generate.subprocess.run", fake_run)

    assert request_install_commands_from_codex(repo_path, "3.12") == expected_commands


def test_codex_install_command_inference_failure_returns_none(tmp_path, monkeypatch):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    monkeypatch.setattr(
        "gso.collect.generate.generate.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="not authenticated"
        ),
    )

    assert request_install_commands_from_codex(repo_path, "3.12") is None


def test_update_yaml_install_commands_replaces_empty_value(tmp_path):
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(
        "# Keep this comment.\n"
        "exp_id: demo\nrepo_url: https://github.com/example/repo\n"
        "install_commands: []\nrepo_instr: |\n  Keep this instruction.\n"
    )
    commands = ["uv venv --python 3.12", "uv pip install -e ."]

    update_yaml_install_commands(yaml_path, commands)

    assert yaml.safe_load(yaml_path.read_text())["install_commands"] == commands
    assert "# Keep this comment." in yaml_path.read_text()
    assert "  Keep this instruction." in yaml_path.read_text()


def test_generator_persists_inferred_commands_to_source_and_experiment_copy(
    tmp_path, monkeypatch
):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    source_path = tmp_path / "source.yaml"
    exp_dir = tmp_path / "exps" / "demo"
    exp_dir.mkdir(parents=True)
    copied_path = exp_dir / "demo.yaml"
    yaml_text = (
        "exp_id: demo\nrepo_url: https://github.com/example/repo\n"
        "py_version: '3.12'\ninstall_commands: []\n"
    )
    source_path.write_text(yaml_text)
    copied_path.write_text(yaml_text)
    commands = ["uv venv --python 3.12", "uv pip install -e . requests"]

    generator = PerfExpGenerator.__new__(PerfExpGenerator)
    generator.config = {
        "exp_id": "demo",
        "repo_url": "https://github.com/example/repo",
        "py_version": "3.12",
        "install_commands": [],
    }
    generator.exp_id = "demo"
    generator.exp_dir = exp_dir
    generator.repo = SimpleNamespace(local_repo_path=repo_path)
    args = SimpleNamespace(yaml_path=str(source_path), docker_repo_path=None)
    monkeypatch.setattr(
        "gso.collect.generate.generate.request_install_commands_from_codex",
        lambda path, py_version: commands,
    )

    generator.ensure_install_commands(args)

    assert generator.config["install_commands"] == commands
    assert yaml.safe_load(source_path.read_text())["install_commands"] == commands
    assert yaml.safe_load(copied_path.read_text())["install_commands"] == commands


def test_atomic_problem_save_preserves_existing_api_commit(tmp_path):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commits = [
        PerformanceCommit(
            commit_hash=f"{index}" * 16,
            subject=f"Commit {index}",
            message=f"Commit {index}",
            date=f"2024-01-0{index}T00:00:00",
        )
        for index in (1, 2)
    ]
    problems = []
    for commit in commits:
        tests = CommitTests.from_commit(commit)
        tests.add_sample(VALID_TEST)
        problems.append(
            Problem(
                pid="repo-target",
                repo=repo,
                api="target",
                commits=[commit],
                tests=[tests],
            )
        )

    path = tmp_path / "repo_problems.json"
    save_problems_atomically(path, [problems[0]])
    save_problems_atomically(path, [problems[1]])

    saved = load_existing_problems(path)
    assert len(saved) == 1
    assert [commit.commit_hash for commit in saved[0].commits] == [
        commit.commit_hash for commit in commits
    ]
    assert [tests.commit_hash for tests in saved[0].tests] == [
        commit.commit_hash for commit in commits
    ]


def test_generated_scenario_and_test_are_parsed_from_one_completion():
    scenario, test = get_generated_scenario_and_test(combined_output())

    assert scenario.title == "scenario"
    assert test == VALID_TEST.strip()

    with pytest.raises(GeneratedScenarioError, match="invalid scenario structure"):
        get_generated_scenario_and_test(combined_output({"title": "incomplete"}))

    with pytest.raises(GeneratedScenarioError, match="exactly two closed fenced"):
        get_generated_scenario_and_test(json.dumps(SCENARIO_TEMPLATE))

    reversed_blocks = (
        f"```python\n{VALID_TEST}\n```\n"
        f"```json\n{json.dumps(SCENARIO_TEMPLATE)}\n```"
    )
    with pytest.raises(GeneratedScenarioError, match="JSON block must appear before"):
        get_generated_scenario_and_test(reversed_blocks)


def test_streaming_runner_uses_one_choice_even_when_test_count_is_larger():
    args = LLMArgs(model_name="custom-model", n=5, use_cache=False)
    model = LanguageModel(
        model_name=args.model_name,
        style=LanguageModelStyle.OpenAI,
    )

    runner = llm_utils.StreamingOpenAIRunner(args, model)

    assert runner.client_kwargs["n"] == 1


def test_runner_applies_configured_extra_body_for_stream_and_non_stream():
    args = LLMArgs(model_name="custom-model", n=1, use_cache=False)
    model = LanguageModel(
        model_name=args.model_name,
        style=LanguageModelStyle.OpenAI,
    )

    for stream in (False, True):
        runner = llm_utils.StreamingOpenAIRunner(
            args,
            model,
            stream=stream,
            extra_body={"enable_thinking": False},
        )
        assert runner.client_kwargs["extra_body"] == {"enable_thinking": False}


def test_runner_omits_extra_body_when_not_configured():
    args = LLMArgs(model_name="custom-model", n=1, use_cache=False)
    model = LanguageModel(
        model_name=args.model_name,
        style=LanguageModelStyle.OpenAI,
    )

    runner = llm_utils.StreamingOpenAIRunner(args, model)

    assert "extra_body" not in runner.client_kwargs


def test_non_stream_runner_requests_one_complete_response(monkeypatch):
    args = LLMArgs(model_name="custom-model", n=1, use_cache=False)
    model = LanguageModel(
        model_name=args.model_name,
        style=LanguageModelStyle.OpenAI,
    )
    runner = llm_utils.StreamingOpenAIRunner(
        args,
        model,
        stream=False,
        extra_body={"enable_thinking": False},
    )
    observed = {}

    def fake_create(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    monkeypatch.setattr(runner, "_get_client", lambda: client)

    assert runner._run_single([{"role": "user", "content": "prompt"}]) == ["answer"]
    assert observed["stream"] is False
    assert observed["extra_body"] == {"enable_thinking": False}


def test_runner_stops_after_five_retries(monkeypatch):
    args = LLMArgs(model_name="custom-model", n=1, use_cache=False)
    model = LanguageModel(
        model_name=args.model_name,
        style=LanguageModelStyle.OpenAI,
    )
    runner = llm_utils.StreamingOpenAIRunner(args, model, stream=False)
    runner.retry_delay_seconds = 0
    attempts = 0

    def fail(_payload):
        nonlocal attempts
        attempts += 1
        raise llm_utils.IncompleteStreamingResponse("empty")

    monkeypatch.setattr(runner, "_consume_non_stream", fail)

    with pytest.raises(llm_utils.IncompleteStreamingResponse, match="empty"):
        runner._run_single([{"role": "user", "content": "prompt"}])

    assert attempts == 6


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

    def fake_completions(request_args, payloads, *, stream):
        observed["n"] = request_args.n
        observed["multiprocess"] = request_args.multiprocess
        observed["payloads"] = payloads
        observed["stream"] = stream
        return [["one choice"]]

    monkeypatch.setattr(llm_utils, "get_llm_completions", fake_completions)

    output = llm_utils.get_streaming_llm_completion(
        args, [{"role": "user", "content": "prompt"}]
    )

    assert output == "one choice"
    assert observed == {
        "n": 1,
        "multiprocess": 1,
        "payloads": [[{"role": "user", "content": "prompt"}]],
        "stream": True,
    }


def test_commit_worker_generates_scenario_and_test_together(monkeypatch):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
        repo_instr="Use repository fixtures.",
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
        commit_tests.init_chat(
            "test system",
            "commit context",
            "write test\n\nRepo-specific Instructions:\nUse repository fixtures.",
        )
        return commit_tests

    responses = iter(
        [
            combined_output({**SCENARIO_TEMPLATE, "title": "first"}),
            combined_output({**SCENARIO_TEMPLATE, "title": "second"}),
        ]
    )
    payloads = []

    def fake_completion(request_args, payload, *, stream):
        assert stream is False
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_llm_completion",
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
    assert len(payloads) == 2
    assert payloads[0][0]["content"] == "test system"
    assert [message["role"] for message in payloads[0]] == ["system", "user"]
    assert payloads[0][1]["content"].startswith("commit context\n\nCreate one benchmark")
    assert "write test" not in payloads[0][1]["content"]
    assert payloads[0][1]["content"].count("Use repository fixtures.") == 1
    assert "Return exactly two fenced blocks" in payloads[0][1]["content"]
    assert "use only APIs and behavior available before the optimization" in (
        payloads[0][1]["content"]
    )
    assert [message["role"] for message in payloads[1]] == ["system", "user"]
    assert '"title":"first"' in payloads[1][1]["content"].replace(" ", "")
    # Duplicate test code is intentionally accepted; no deduplication is performed.
    assert len(result.test_outputs) == 2
    assert result.commit_tests.samples[0].startswith("# GSO generated scenario:")
    assert "# title: first" in result.commit_tests.samples[0]
    assert "# title: second" in result.commit_tests.samples[1]


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

    invalid_scenario = (
        '```json\n{"title": "broken", "workload": unquoted}\n```\n'
        f"```python\n{VALID_TEST}\n```"
    )
    responses = iter([invalid_scenario, combined_output()])
    calls = []

    def fake_completion(request_args, payload, *, stream):
        assert stream is False
        calls.append((request_args, payload))
        return next(responses)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_llm_completion",
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
    assert len(calls) == 2
    assert calls[0][0].use_cache is True
    assert calls[1][0].use_cache is False
    assert [message["role"] for message in calls[1][1]] == ["system", "user"]
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

    invalid_test = "```python\nimport timeit\n```"
    invalid_output = f"```json\n{json.dumps(SCENARIO_TEMPLATE)}\n```\n{invalid_test}"
    responses = iter([invalid_output, combined_output()])
    calls = []

    def fake_completion(request_args, payload, *, stream):
        assert stream is False
        calls.append((request_args, payload))
        return next(responses)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_llm_completion",
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
    assert result.test_outputs == [invalid_output, combined_output()]
    assert calls[0][0].use_cache is True
    assert calls[1][0].use_cache is False
    retry_instruction = calls[1][1][-1]["content"]
    assert "semantic retry 1 of 3" in retry_instruction
    assert "missing required function" in retry_instruction


def test_commit_worker_retries_failed_execution_and_records_reason(monkeypatch, capsys):
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

    responses = iter([combined_output(), combined_output()])
    completion_calls = []
    execution_calls = []

    def fake_completion(request_args, payload, *, stream):
        assert stream is False
        completion_calls.append((request_args, payload))
        return next(responses)

    def fake_evaluate(problem_arg, commit_arg, test_arg, config_arg):
        execution_calls.append(test_arg)
        if len(execution_calls) == 1:
            return GeneratedTestExecutionResult(
                passed=False,
                error="AssertionError: candidate output differs from reference",
                log_path="/tmp/test-execution.log",
            )
        return GeneratedTestExecutionResult(passed=True)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_llm_completion",
        fake_completion,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.evaluate_generated_test", fake_evaluate
    )

    result = generate_commit_tests(
        CommitGenerationTask(
            problem_index=0,
            commit_index=0,
            repo=repo,
            problem=problem,
            commit=commit,
            args=args,
            execution_config=GeneratedTestExecutionConfig(exp_id="repo"),
        )
    )

    assert result.error is None
    assert len(execution_calls) == 2
    assert result.test_attempts[0]["error"].endswith(
        "AssertionError: candidate output differs from reference"
    )
    assert result.test_attempts[0]["execution_log"] == "/tmp/test-execution.log"
    assert result.test_attempts[1]["error"] is None
    retry_instruction = completion_calls[1][1][-1]["content"]
    assert "semantic retry 1 of 3" in retry_instruction
    assert "candidate output differs from reference" in retry_instruction
    output_lines = [
        re.sub(r"(?<==)\d+\.\d+s", "<seconds>", line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert output_lines == [
        (
            f"{GENERATION_EVENT_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/1 | requesting test"
        ),
        (
            f"{GENERATION_EVENT_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/1 | starting test (attempt 1/4)"
        ),
        (
            f"{GENERATION_EVENT_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/1 | test failed (log: /tmp/test-execution.log)"
        ),
        (
            f"{GENERATION_EVENT_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/1 | re-requesting test (semantic retry 1/3)"
        ),
        (
            f"{GENERATION_EVENT_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/1 | starting test (attempt 2/4)"
        ),
        (
            f"{GENERATION_EVENT_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/1 | test passed"
        ),
        (
            f"{GENERATION_PROGRESS_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/1 accepted | accepted=1 | skipped=0 | "
            "semantic_retries=1 | elapsed=<seconds> | llm=<seconds> | "
            "validation=<seconds> | slots complete"
        ),
    ]


def test_commit_worker_skips_failed_test_slot_and_continues(monkeypatch, capsys):
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
    args = PerfExpGenArgs(yaml_path="unused.yaml", model_name="custom-model", n=2)

    def fake_prepare(task_args):
        _, _, prepared_commit, _ = task_args
        commit_tests = CommitTests.from_commit(prepared_commit)
        commit_tests.init_chat("test system", "commit context", "write test")
        return commit_tests

    responses = iter(
        [
            *([combined_output()] * (SEMANTIC_RETRY_COUNT + 1)),
            combined_output({**SCENARIO_TEMPLATE, "title": "second"}),
        ]
    )
    execution_count = 0

    def fake_evaluate(*args):
        nonlocal execution_count
        execution_count += 1
        if execution_count <= SEMANTIC_RETRY_COUNT + 1:
            return GeneratedTestExecutionResult(passed=False, error="always fails")
        return GeneratedTestExecutionResult(passed=True)

    monkeypatch.setattr("gso.collect.generate.generate.prepare_mp_helper", fake_prepare)
    monkeypatch.setattr(
        "gso.collect.generate.generate.get_llm_completion",
        lambda request_args, payload, *, stream: next(responses),
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.evaluate_generated_test", fake_evaluate
    )

    result = generate_commit_tests(
        CommitGenerationTask(
            problem_index=0,
            commit_index=0,
            repo=repo,
            problem=problem,
            commit=commit,
            args=args,
            execution_config=GeneratedTestExecutionConfig(exp_id="repo"),
        )
    )

    assert result.error is None
    assert result.commit_tests is not None
    assert result.commit_tests.num_samples() == 1
    assert execution_count == SEMANTIC_RETRY_COUNT + 2
    assert len(result.test_attempts) == SEMANTIC_RETRY_COUNT + 2
    assert result.failed_test_slots == [
        {
            "scenario_index": 1,
            "error": (
                "repo.target/abcdef1 scenario/test 1: automated execution failed:\n"
                "always fails"
            ),
            "execution_log": None,
        }
    ]
    assert result.test_attempts[-1]["scenario_index"] == 2
    assert result.test_attempts[-1]["error"] is None
    assert "# title: second" in result.commit_tests.samples[0]
    output_lines = [
        re.sub(r"(?<==)\d+\.\d+s", "<seconds>", line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert [
        line for line in output_lines if line.startswith(GENERATION_PROGRESS_PREFIX)
    ] == [
        (
            f"{GENERATION_PROGRESS_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 1/2 skipped | accepted=0 | skipped=1 | semantic_retries=3 | "
            "elapsed=<seconds> | llm=<seconds> | validation=<seconds>"
        ),
        (
            f"{GENERATION_PROGRESS_PREFIX} target 1/1 | repo.target/abcdef1 | "
            "test 2/2 accepted | accepted=1 | skipped=1 | "
            "semantic_retries=3 | elapsed=<seconds> | llm=<seconds> | "
            "validation=<seconds> | slots complete"
        ),
    ]


def test_generator_skips_failed_commit_and_saves_successful_commit(
    monkeypatch, tmp_path
):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    failed_commit = PerformanceCommit(
        commit_hash="1111111234567890",
        subject="Broken generation",
        message="Broken generation",
        date="2024-01-01T00:00:00",
    )
    successful_commit = PerformanceCommit(
        commit_hash="2222222234567890",
        subject="Successful generation",
        message="Successful generation",
        date="2024-01-02T00:00:00",
    )
    generator = object.__new__(PerfExpGenerator)
    generator.config = {"py_version": "3.12"}
    generator.repo = repo
    generator.candidates = {"target": [failed_commit, successful_commit]}
    generator.exp_id = "repo"
    generator.exp_dir = tmp_path
    args = PerfExpGenArgs(yaml_path="unused.yaml", n=1, multiprocess=1)

    successful_tests = CommitTests.from_commit(successful_commit)
    successful_tests.add_sample(VALID_TEST)
    outputs = [
        SimpleNamespace(
            is_success=lambda: True,
            result=CommitGenerationResult(
                problem_index=0,
                commit_index=0,
                pid="repo-target",
                commit_hash=failed_commit.commit_hash,
                error="commit setup failed",
            ),
        ),
        SimpleNamespace(
            is_success=lambda: True,
            result=CommitGenerationResult(
                problem_index=0,
                commit_index=1,
                pid="repo-target",
                commit_hash=successful_commit.commit_hash,
                commit_tests=successful_tests,
            ),
        ),
    ]
    saved = {}
    parallel_call = {}

    def fake_run_tasks(tasks, num_workers):
        parallel_call.update(tasks=tasks, num_workers=num_workers)
        return zip(tasks, outputs)

    monkeypatch.setattr(
        "gso.collect.generate.generate.prepare_generated_test_execution",
        lambda problems, config: None,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.generate_commits_as_completed",
        fake_run_tasks,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.validate_problem_test_samples",
        lambda problems: None,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.save_problems_atomically",
        lambda path, problems: saved.update(path=path, problems=problems),
    )
    monkeypatch.setattr(
        generator,
        "save_retry_diagnostics",
        lambda diagnostics, args: saved.update(diagnostics=diagnostics),
    )

    problems = generator.gen(args)

    assert problems == saved["problems"]
    assert len(problems) == 1
    assert problems[0].commits == [successful_commit]
    assert problems[0].tests == [successful_tests]
    assert saved["path"] == tmp_path / "repo_problems.json"
    assert [task.target_index for task in parallel_call["tasks"]] == [0, 1]
    assert [task.target_count for task in parallel_call["tasks"]] == [2, 2]
    assert parallel_call["num_workers"] == 1
    assert len(saved["diagnostics"]) == 1
    assert saved["diagnostics"][0]["commit_hash"] == failed_commit.commit_hash
    assert saved["diagnostics"][0]["error"] == "commit setup failed"


def test_generator_reuses_existing_api_commit_and_only_generates_missing_commit(
    monkeypatch, tmp_path
):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    existing_commit = PerformanceCommit(
        commit_hash="1111111234567890",
        subject="Already generated",
        message="Already generated",
        date="2024-01-01T00:00:00",
    )
    missing_commit = PerformanceCommit(
        commit_hash="2222222234567890",
        subject="Needs generation",
        message="Needs generation",
        date="2024-01-02T00:00:00",
    )
    existing_tests = CommitTests.from_commit(existing_commit)
    existing_tests.add_sample(VALID_TEST)
    results_path = tmp_path / "repo_problems.json"
    save_problems_atomically(
        results_path,
        [
            Problem(
                pid="repo-target",
                repo=repo,
                api="target",
                commits=[existing_commit],
                tests=[existing_tests],
            )
        ],
    )

    generator = object.__new__(PerfExpGenerator)
    generator.config = {"py_version": "3.12"}
    generator.repo = repo
    generator.candidates = {"target": [existing_commit, missing_commit]}
    generator.exp_id = "repo"
    generator.exp_dir = tmp_path
    args = PerfExpGenArgs(yaml_path="unused.yaml", n=1, multiprocess=2)

    missing_tests = CommitTests.from_commit(missing_commit)
    missing_tests.add_sample(VALID_TEST)
    generated_tasks = []

    def fake_run_tasks(tasks, num_workers):
        generated_tasks.extend(tasks)
        assert num_workers == 1
        yield tasks[0], SimpleNamespace(
            is_success=lambda: True,
            result=CommitGenerationResult(
                problem_index=0,
                commit_index=0,
                pid="repo-target",
                commit_hash=missing_commit.commit_hash,
                commit_tests=missing_tests,
            ),
        )

    monkeypatch.setattr(
        "gso.collect.generate.generate.prepare_generated_test_execution",
        lambda problems, config: None,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.generate_commits_as_completed",
        fake_run_tasks,
    )
    monkeypatch.setattr(
        generator,
        "save_retry_diagnostics",
        lambda diagnostics, args: None,
    )

    problems = generator.gen(args)

    assert [task.commit.commit_hash for task in generated_tasks] == [
        missing_commit.commit_hash
    ]
    assert len(problems) == 1
    assert [commit.commit_hash for commit in problems[0].commits] == [
        existing_commit.commit_hash,
        missing_commit.commit_hash,
    ]
    persisted = load_existing_problems(results_path)
    assert [commit.commit_hash for commit in persisted[0].commits] == [
        existing_commit.commit_hash,
        missing_commit.commit_hash,
    ]


def test_generator_does_not_prepare_docker_when_all_api_commits_exist(
    monkeypatch, tmp_path
):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commit = PerformanceCommit(
        commit_hash="1111111234567890",
        subject="Already generated",
        message="Already generated",
        date="2024-01-01T00:00:00",
    )
    tests = CommitTests.from_commit(commit)
    tests.add_sample(VALID_TEST)
    existing_problem = Problem(
        pid="repo-target",
        repo=repo,
        api="target",
        commits=[commit],
        tests=[tests],
    )
    save_problems_atomically(tmp_path / "repo_problems.json", [existing_problem])

    generator = object.__new__(PerfExpGenerator)
    generator.config = {"py_version": "3.12"}
    generator.repo = repo
    generator.candidates = {"target": [commit]}
    generator.exp_id = "repo"
    generator.exp_dir = tmp_path
    args = PerfExpGenArgs(yaml_path="unused.yaml", n=1, multiprocess=1)

    monkeypatch.setattr(
        "gso.collect.generate.generate.prepare_generated_test_execution",
        lambda *args: pytest.fail("Docker preparation should be skipped"),
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.generate_commits_as_completed",
        lambda *args: pytest.fail("Generation should be skipped"),
    )

    assert generator.gen(args) == [existing_problem]


def test_generator_persists_each_commit_before_receiving_the_next(
    monkeypatch, tmp_path
):
    repo = Repo(
        repo_url="https://github.com/example/repo",
        repo_owner="example",
        repo_name="repo",
    )
    commits = [
        PerformanceCommit(
            commit_hash="1111111234567890",
            subject="First generation",
            message="First generation",
            date="2024-01-01T00:00:00",
        ),
        PerformanceCommit(
            commit_hash="2222222234567890",
            subject="Second generation",
            message="Second generation",
            date="2024-01-02T00:00:00",
        ),
    ]
    generator = object.__new__(PerfExpGenerator)
    generator.config = {"py_version": "3.12"}
    generator.repo = repo
    generator.candidates = {"target": commits}
    generator.exp_id = "repo"
    generator.exp_dir = tmp_path
    args = PerfExpGenArgs(yaml_path="unused.yaml", n=1, multiprocess=2)

    outputs = []
    for commit_index, commit in enumerate(commits):
        tests = CommitTests.from_commit(commit)
        tests.add_sample(VALID_TEST)
        outputs.append(
            SimpleNamespace(
                is_success=lambda: True,
                result=CommitGenerationResult(
                    problem_index=0,
                    commit_index=commit_index,
                    pid="repo-target",
                    commit_hash=commit.commit_hash,
                    commit_tests=tests,
                ),
            )
        )

    saved_commit_hashes = []

    def fake_save(_path, problems):
        saved_commit_hashes.append(
            [commit.commit_hash for problem in problems for commit in problem.commits]
        )

    def fake_run_tasks(tasks, _num_workers):
        yield tasks[0], outputs[0]
        assert saved_commit_hashes == [[commits[0].commit_hash]]
        yield tasks[1], outputs[1]

    monkeypatch.setattr(
        "gso.collect.generate.generate.prepare_generated_test_execution",
        lambda problems, config: None,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.generate_commits_as_completed",
        fake_run_tasks,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.validate_problem_test_samples",
        lambda problems: None,
    )
    monkeypatch.setattr(
        "gso.collect.generate.generate.save_problems_atomically",
        fake_save,
    )
    monkeypatch.setattr(
        generator,
        "save_retry_diagnostics",
        lambda diagnostics, args: None,
    )

    problems = generator.gen(args)

    assert saved_commit_hashes == [
        [commits[0].commit_hash],
        [commit.commit_hash for commit in commits],
    ]
    assert problems[0].commits == commits
