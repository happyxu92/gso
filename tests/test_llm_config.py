from types import SimpleNamespace

from gso.collect.analysis import commits
from gso.collect.analysis import retriever as retriever_module
from gso.collect.generate.args import PerfExpGenArgs
from gso.collect.generate.generate import PerfExpGenerator
from gso.data import PerformanceCommit
from gso.utils.llm import ConfiguredLLM, configure_openai_compatible_llm


def test_llm_config_parses_request_limits_without_defaults(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    configured = configure_openai_compatible_llm(
        {
            "llm": {
                "model_name": "test-openai-compatible-model",
                "api_key_env": "LLM_API_KEY",
                "max_tokens": 32768,
                "openai_timeout": 600,
            }
        },
        default_model="default-model",
        default_multiprocess=1,
        purpose="test",
    )

    assert configured.max_tokens == 32768
    assert configured.openai_timeout == 600


def test_commit_analysis_applies_yaml_limits_to_every_llm_stage(monkeypatch):
    config = {
        "llm": {
            "model_name": "GLM-5.2",
            "multiprocess": 1,
            "max_tokens": 32768,
            "openai_timeout": 600,
        }
    }

    monkeypatch.setattr(
        commits,
        "configure_openai_compatible_llm",
        lambda *args, **kwargs: ConfiguredLLM(
            model_name="GLM-5.2",
            multiprocess=1,
            base_url="https://antchat.alipay.com/v1",
            max_tokens=32768,
            openai_timeout=600,
        ),
    )

    commits.PerfCommitAnalyzer.configure_llm(config)

    for cache_stage, default_max_tokens in (
        ("commit_filter", 10000),
        ("affected_files", 24000),
        ("api_identification", 24000),
    ):
        args = commits.PerfCommitAnalyzer.build_llm_args(
            cache_stage=cache_stage,
            default_max_tokens=default_max_tokens,
        )
        assert args.max_tokens == 32768
        assert args.openai_timeout == 600


def test_generation_applies_yaml_request_limits(monkeypatch):
    generator = object.__new__(PerfExpGenerator)
    generator.config = {
        "llm": {
            "model_name": "GLM-5.2",
            "multiprocess": 1,
            "max_tokens": 32768,
            "openai_timeout": 600,
        }
    }
    args = PerfExpGenArgs(yaml_path="experiment.yaml")

    monkeypatch.setattr(
        "gso.collect.generate.generate.configure_openai_compatible_llm",
        lambda *args, **kwargs: ConfiguredLLM(
            model_name="GLM-5.2",
            multiprocess=1,
            base_url="https://antchat.alipay.com/v1",
            max_tokens=32768,
            openai_timeout=600,
        ),
    )

    generator.configure_llm(args)

    assert args.max_tokens == 32768
    assert args.openai_timeout == 600


def _performance_commit() -> PerformanceCommit:
    return PerformanceCommit(
        commit_hash="abcdef1234567890",
        subject="Improve performance",
        message="Improve performance",
        date="2024-01-01T00:00:00",
        diff_text="diff --git a/module.py b/module.py",
    )


def _configure_analysis_limits(monkeypatch) -> None:
    analyzer = commits.PerfCommitAnalyzer
    monkeypatch.setattr(analyzer, "model_name", "GLM-5.2")
    monkeypatch.setattr(analyzer, "llm_multiprocess", 1)
    monkeypatch.setattr(analyzer, "llm_max_tokens", 32768)
    monkeypatch.setattr(analyzer, "llm_openai_timeout", 600)


def test_commit_filter_uses_streaming_completions(monkeypatch, tmp_path):
    _configure_analysis_limits(monkeypatch)
    commit = _performance_commit()
    observed = {}

    def fake_streaming(args, prompts):
        observed["args"] = args
        observed["prompts"] = prompts
        return [['{"reason": "faster", "answer": "yes"}']]

    monkeypatch.setattr(commits, "get_streaming_llm_completions", fake_streaming)
    monkeypatch.setattr(
        commits.PerfCommitAnalyzer,
        "retrieve_affected_files",
        lambda filtered, repo_path: SimpleNamespace(),
    )

    filtered, _ = commits.PerfCommitAnalyzer.llm_analysis([commit], tmp_path)

    assert filtered == [commit]
    assert observed["args"].max_tokens == 32768
    assert observed["args"].openai_timeout == 600
    assert len(observed["prompts"]) == 1


def test_affected_files_uses_streaming_completions(monkeypatch):
    _configure_analysis_limits(monkeypatch)
    commit = _performance_commit()
    retriever = object.__new__(retriever_module.Retriever)
    retriever.build_prompt = lambda candidate: [
        {"role": "user", "content": candidate.message}
    ]
    retriever.extract_match_file_names = lambda response: [response]
    observed = {}

    def fake_streaming(args, prompts):
        observed["args"] = args
        observed["prompts"] = prompts
        return [["module.py"]]

    monkeypatch.setattr(
        retriever_module, "get_streaming_llm_completions", fake_streaming
    )
    args = commits.PerfCommitAnalyzer.build_llm_args(
        cache_stage="affected_files", default_max_tokens=24000
    )

    retriever.retrieve_affected_files([commit], args)

    assert commit.affected_paths == ["module.py"]
    assert observed["args"].max_tokens == 32768
    assert observed["args"].openai_timeout == 600
    assert len(observed["prompts"]) == 1


def test_api_identification_uses_streaming_completions(monkeypatch):
    _configure_analysis_limits(monkeypatch)
    commit = _performance_commit()
    commit.add_affected_paths([])
    retriever = SimpleNamespace(file_content_map={})
    observed = {}

    def fake_streaming(args, prompts):
        observed["args"] = args
        observed["prompts"] = prompts
        return [['{"reason": "public entry point", "apis": ["module.api"]}']]

    monkeypatch.setattr(commits, "get_streaming_llm_completions", fake_streaming)

    commits.PerfCommitAnalyzer.llm_get_apis([commit], retriever)

    assert commit.apis == ["module.api"]
    assert observed["args"].max_tokens == 32768
    assert observed["args"].openai_timeout == 600
    assert len(observed["prompts"]) == 1
