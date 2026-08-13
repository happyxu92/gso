import os
from dataclasses import dataclass
from time import sleep
from typing import Any, ClassVar

import httpx
import openai
from openai import OpenAI
from r2e.llms.base_runner import BaseRunner
from r2e.llms.language_model import (
    LanguageModel,
    LanguageModelList,
    LanguageModelStyle,
)


@dataclass(frozen=True)
class ConfiguredLLM:
    model_name: str
    multiprocess: int
    base_url: str | None
    max_tokens: int | None = None
    openai_timeout: int | None = None


class IncompleteStreamingResponse(RuntimeError):
    """Raised when a stream does not produce every requested text completion."""


class StreamingOpenAIRunner(BaseRunner):
    """R2E-compatible OpenAI runner that aggregates streamed completion chunks."""

    client: ClassVar[OpenAI | None] = None
    retry_delay_seconds = 30

    def __init__(self, args, model: LanguageModel):
        super().__init__(args, model)
        if any(name in args.model_name for name in ("o1", "o3", "o4")):
            self.client_kwargs: dict[str, Any] = {
                "model": args.model_name,
                "max_completion_tokens": args.max_tokens,
            }
        else:
            self.client_kwargs = {
                "model": args.model_name,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
                "n": 1,
                "timeout": args.openai_timeout,
            }

    def config(self) -> dict[str, Any]:
        # Include the transport mode in R2E's cache key.
        return {**self.client_kwargs, "stream": True}

    def _get_client(self) -> OpenAI:
        if StreamingOpenAIRunner.client is None:
            api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self.args.openai_timeout,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            StreamingOpenAIRunner.client = OpenAI(**client_kwargs)
        return StreamingOpenAIRunner.client

    def _consume_stream(self, payload: list[dict[str, str]]) -> list[str]:
        expected_choices = int(self.client_kwargs.get("n", 1))
        chunks_by_choice: dict[int, list[str]] = {
            index: [] for index in range(expected_choices)
        }
        observed_choices: set[int] = set()

        stream = self._get_client().chat.completions.create(
            messages=payload,
            stream=True,
            **self.client_kwargs,
        )
        try:
            for chunk in stream:
                for choice in chunk.choices:
                    index = choice.index
                    if index not in chunks_by_choice:
                        raise IncompleteStreamingResponse(
                            f"stream returned unexpected choice index {index}; "
                            f"expected 0..{expected_choices - 1}"
                        )
                    observed_choices.add(index)
                    if choice.delta.content:
                        chunks_by_choice[index].append(choice.delta.content)
        finally:
            stream.close()

        expected_indices = set(chunks_by_choice)
        if observed_choices != expected_indices:
            missing = sorted(expected_indices - observed_choices)
            raise IncompleteStreamingResponse(
                f"stream ended before returning choice index(es): {missing}"
            )

        outputs = [
            "".join(chunks_by_choice[index]) for index in range(expected_choices)
        ]
        empty_choices = [
            index for index, output in enumerate(outputs) if not output.strip()
        ]
        if empty_choices:
            raise IncompleteStreamingResponse(
                f"stream ended without content for choice index(es): {empty_choices}"
            )
        return outputs

    def _run_single(self, payload: list[dict[str, str]]) -> list[str]:
        assert isinstance(payload, list)

        while True:
            try:
                return self._consume_stream(payload)
            except (
                openai.OpenAIError,
                httpx.TransportError,
                IncompleteStreamingResponse,
            ) as error:
                print("Exception: ", repr(error), flush=True)
                cause = error.__cause__ or error.__context__
                if cause is not None:
                    print("Caused by: ", repr(cause), flush=True)
                print(
                    f"Sleeping for {self.retry_delay_seconds} seconds...",
                    flush=True,
                )
                print(
                    "Consider reducing the number of parallel processes.",
                    flush=True,
                )
                sleep(self.retry_delay_seconds)


def _get_openai_model(model_name: str) -> LanguageModel:
    """Resolve a configured model, registering custom models in spawned workers."""
    matched_models = [
        model for model in LanguageModelList if model.model_name == model_name
    ]
    if not matched_models:
        model = LanguageModel(
            model_name=model_name,
            style=LanguageModelStyle.OpenAI,
        )
        LanguageModelList.append(model)
        return model
    if len(matched_models) != 1:
        raise ValueError(
            f"Expected exactly one language model named {model_name!r}, "
            f"found {len(matched_models)}"
        )
    return matched_models[0]


def get_streaming_llm_completions(args, payloads: list) -> list[list[str]]:
    """Generate one streamed choice for each OpenAI-compatible payload."""
    request_args = args.model_copy(update={"n": 1})
    model = _get_openai_model(request_args.model_name)
    if model.style != LanguageModelStyle.OpenAI:
        raise ValueError(f"Streaming is unsupported for model style: {model.style}")

    return StreamingOpenAIRunner(request_args, model).run_main(payloads)


def get_streaming_llm_completion(args, payload: list[dict[str, str]]) -> str:
    """Generate exactly one choice without introducing payload-level concurrency."""
    request_args = args.model_copy(update={"n": 1, "multiprocess": 1})
    outputs = get_streaming_llm_completions(request_args, [payload])
    if len(outputs) != 1 or len(outputs[0]) != 1:
        choice_counts = [len(group) for group in outputs]
        raise IncompleteStreamingResponse(
            "Expected one completion group containing one choice, "
            f"got {len(outputs)} group(s) with choice count(s) {choice_counts}"
        )
    return outputs[0][0]


def configure_openai_compatible_llm(
    config: dict,
    *,
    default_model: str,
    default_multiprocess: int,
    default_max_tokens: int | None = None,
    default_openai_timeout: int | None = None,
    model_env: str | None = None,
    purpose: str = "LLM",
) -> ConfiguredLLM:
    """Configure R2E for OpenAI or an OpenAI-compatible endpoint."""
    llm_config = config.get("llm", {}) or {}
    if not isinstance(llm_config, dict):
        raise ValueError("The 'llm' experiment setting must be a YAML mapping")

    model_name = (
        llm_config.get("model_name")
        or (os.getenv(model_env) if model_env else None)
        or default_model
    )
    base_url = llm_config.get("base_url") or os.getenv("OPENAI_BASE_URL")
    api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("llm.model_name must be a non-empty string")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("llm.api_key_env must be a non-empty string")

    api_key_env = api_key_env.strip()
    api_key = (
        os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    )
    if not api_key:
        raise ValueError(
            f"Missing LLM API key. Set the '{api_key_env}' environment variable. "
            "For a local endpoint that does not authenticate, set it to a dummy value."
        )

    try:
        multiprocess = int(llm_config.get("multiprocess", default_multiprocess))
    except (TypeError, ValueError) as exc:
        raise ValueError("llm.multiprocess must be an integer") from exc
    if multiprocess < 1:
        raise ValueError("llm.multiprocess must be at least 1")

    max_tokens = None
    if default_max_tokens is not None:
        raw_max_tokens = llm_config.get("max_tokens", default_max_tokens)
        if isinstance(raw_max_tokens, bool):
            raise ValueError("llm.max_tokens must be a positive integer")
        if isinstance(raw_max_tokens, int):
            max_tokens = raw_max_tokens
        elif isinstance(raw_max_tokens, str) and raw_max_tokens.strip().isdigit():
            max_tokens = int(raw_max_tokens.strip())
        else:
            raise ValueError("llm.max_tokens must be a positive integer")
        if max_tokens < 1:
            raise ValueError("llm.max_tokens must be at least 1")

    openai_timeout = None
    if default_openai_timeout is not None:
        raw_openai_timeout = llm_config.get("openai_timeout", default_openai_timeout)
        if isinstance(raw_openai_timeout, bool):
            raise ValueError("llm.openai_timeout must be a positive integer")
        if isinstance(raw_openai_timeout, int):
            openai_timeout = raw_openai_timeout
        elif (
            isinstance(raw_openai_timeout, str) and raw_openai_timeout.strip().isdigit()
        ):
            openai_timeout = int(raw_openai_timeout.strip())
        else:
            raise ValueError("llm.openai_timeout must be a positive integer")
        if openai_timeout < 1:
            raise ValueError("llm.openai_timeout must be at least 1")

    model_name = model_name.strip()
    base_url = str(base_url).rstrip("/") if base_url else None

    # R2E initializes its OpenAI client at import time. Set standard variables
    # first so api_key_env may refer to any dotenv variable.
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    from r2e.llms.openai_runner import OpenAIRunner

    if not any(model.model_name == model_name for model in LanguageModelList):
        LanguageModelList.append(
            LanguageModel(
                model_name=model_name,
                style=LanguageModelStyle.OpenAI,
            )
        )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if openai_timeout is not None:
        client_kwargs["timeout"] = openai_timeout
    client = OpenAI(**client_kwargs)
    OpenAIRunner.client = client
    StreamingOpenAIRunner.client = client

    endpoint = base_url or "https://api.openai.com/v1"
    settings = []
    if max_tokens is not None:
        settings.append(f"max_tokens={max_tokens}")
    if openai_timeout is not None:
        settings.append(f"openai_timeout={openai_timeout}s")
    settings_info = f" with {', '.join(settings)}" if settings else ""
    print(f"Using {purpose} model '{model_name}' via {endpoint}{settings_info}")
    return ConfiguredLLM(
        model_name=model_name,
        multiprocess=multiprocess,
        base_url=base_url,
        max_tokens=max_tokens,
        openai_timeout=openai_timeout,
    )
