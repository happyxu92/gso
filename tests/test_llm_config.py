from types import SimpleNamespace

import pytest

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
                "stream": True,
                "extra_body": {"enable_thinking": False},
            }
        },
        default_model="default-model",
        default_multiprocess=1,
        purpose="test",
    )

    assert configured.max_tokens == 32768
    assert configured.openai_timeout == 600
    assert configured.stream is True
    assert configured.extra_body == {"enable_thinking": False}


def test_llm_config_defaults_to_non_stream(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    configured = configure_openai_compatible_llm(
        {"llm": {"api_key_env": "LLM_API_KEY"}},
        default_model="default-model",
        default_multiprocess=1,
        purpose="test",
    )

    assert configured.stream is False
    assert configured.extra_body is None


def test_llm_config_rejects_non_boolean_stream(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with pytest.raises(ValueError, match="llm.stream must be a boolean"):
        configure_openai_compatible_llm(
            {"llm": {"api_key_env": "LLM_API_KEY", "stream": "false"}},
            default_model="default-model",
            default_multiprocess=1,
            purpose="test",
        )


def test_llm_config_rejects_non_mapping_extra_body(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with pytest.raises(ValueError, match="llm.extra_body must be a YAML mapping"):
        configure_openai_compatible_llm(
            {"llm": {"api_key_env": "LLM_API_KEY", "extra_body": False}},
            default_model="default-model",
            default_multiprocess=1,
            purpose="test",
        )


def test_commit_analysis_applies_yaml_limits_to_every_llm_stage(monkeypatch):
    config = {
        "llm": {
            "model_name": "GLM-5.2",
            "multiprocess": 1,
            "max_tokens": 32768,
            "openai_timeout": 600,
            "stream": True,
            "extra_body": {"enable_thinking": False},
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
            stream=True,
            extra_body={"enable_thinking": False},
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
    assert commits.PerfCommitAnalyzer.llm_stream is True
    assert commits.PerfCommitAnalyzer.llm_extra_body == {"enable_thinking": False}


def test_generation_applies_yaml_request_limits(monkeypatch):
    generator = object.__new__(PerfExpGenerator)
    generator.config = {
        "llm": {
            "model_name": "GLM-5.2",
            "multiprocess": 1,
            "max_tokens": 32768,
            "openai_timeout": 600,
            "stream": True,
            "extra_body": {"enable_thinking": False},
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
            stream=True,
            extra_body={"enable_thinking": False},
        ),
    )

    generator.configure_llm(args)

    assert args.max_tokens == 32768
    assert args.openai_timeout == 600
    assert args.stream is True
    assert args.extra_body == {"enable_thinking": False}


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
    monkeypatch.setattr(analyzer, "llm_stream", False)
    monkeypatch.setattr(analyzer, "llm_extra_body", {"enable_thinking": False})


def test_commit_filter_defaults_to_non_stream_completions(monkeypatch, tmp_path):
    _configure_analysis_limits(monkeypatch)
    commit = _performance_commit()
    observed = {}

    def fake_completions(args, prompts, *, stream, extra_body):
        observed["args"] = args
        observed["prompts"] = prompts
        observed["stream"] = stream
        observed["extra_body"] = extra_body
        return [['{"reason": "faster", "answer": "yes"}']]

    monkeypatch.setattr(commits, "get_llm_completions", fake_completions)
    monkeypatch.setattr(
        commits.PerfCommitAnalyzer,
        "retrieve_affected_files",
        lambda filtered, repo_path: SimpleNamespace(),
    )

    filtered, _ = commits.PerfCommitAnalyzer.llm_analysis([commit], tmp_path)

    assert filtered == [commit]
    assert observed["args"].max_tokens == 32768
    assert observed["args"].openai_timeout == 600
    assert observed["stream"] is False
    assert observed["extra_body"] == {"enable_thinking": False}
    assert len(observed["prompts"]) == 1


def test_affected_files_defaults_to_non_stream_completions(monkeypatch):
    _configure_analysis_limits(monkeypatch)
    commit = _performance_commit()
    retriever = object.__new__(retriever_module.Retriever)
    retriever.build_prompt = lambda candidate: [
        {"role": "user", "content": candidate.message}
    ]
    retriever.extract_match_file_names = lambda response: [response]
    observed = {}

    def fake_completions(args, prompts, *, stream, extra_body):
        observed["args"] = args
        observed["prompts"] = prompts
        observed["stream"] = stream
        observed["extra_body"] = extra_body
        return [["module.py"]]

    monkeypatch.setattr(retriever_module, "get_llm_completions", fake_completions)
    args = commits.PerfCommitAnalyzer.build_llm_args(
        cache_stage="affected_files", default_max_tokens=24000
    )

    retriever.retrieve_affected_files(
        [commit], args, extra_body=commits.PerfCommitAnalyzer.llm_extra_body
    )

    assert commit.affected_paths == ["module.py"]
    assert observed["args"].max_tokens == 32768
    assert observed["args"].openai_timeout == 600
    assert observed["stream"] is False
    assert observed["extra_body"] == {"enable_thinking": False}
    assert len(observed["prompts"]) == 1


def test_api_identification_defaults_to_non_stream_completions(monkeypatch):
    _configure_analysis_limits(monkeypatch)
    commit = _performance_commit()
    commit.add_affected_paths([])
    retriever = SimpleNamespace(file_content_map={})
    observed = {}

    def fake_completions(args, prompts, *, stream, extra_body):
        observed["args"] = args
        observed["prompts"] = prompts
        observed["stream"] = stream
        observed["extra_body"] = extra_body
        return [['{"reason": "public entry point", "apis": ["module.api"]}']]

    monkeypatch.setattr(commits, "get_llm_completions", fake_completions)

    commits.PerfCommitAnalyzer.llm_get_apis([commit], retriever)

    assert commit.apis == ["module.api"]
    assert observed["args"].max_tokens == 32768
    assert observed["args"].openai_timeout == 600
    assert observed["stream"] is False
    assert observed["extra_body"] == {"enable_thinking": False}
    assert len(observed["prompts"]) == 1
