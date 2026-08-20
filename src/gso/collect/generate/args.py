from typing import Any

from r2e.llms.llm_args import LLMArgs
from pydantic import Field


DEFAULT_GENERATION_MAX_TOKENS = 32768
DEFAULT_GENERATION_OPENAI_TIMEOUT = 600
DEFAULT_GENERATION_DOCKER_BASE_IMAGE = "gso-base:ubuntu22.04-py312-uv0.5.4-amd64"


class PerfExpGenArgs(LLMArgs):
    yaml_path: str = Field(..., description="Path to the experiment YAML file.")
    model_name: str = Field("gpt-4o", description="Model name.")
    cache_batch_size: int = Field(100, description="Batch size for caching.")
    max_year: int = Field(2016, description="Maximum year for commits.")
    min_loc: int = Field(0, description="Minimum LOC for a commit.")
    multiprocess: int = Field(
        4, description="Number of commits generated concurrently."
    )
    n: int = Field(
        5,
        ge=1,
        description=(
            "Number of scenarios/tests generated per commit. Each scenario is "
            "planned, generated, and executed before the next one is requested."
        ),
    )
    max_tokens: int = Field(
        DEFAULT_GENERATION_MAX_TOKENS,
        ge=1,
        description="Maximum output tokens per generated performance test.",
    )
    openai_timeout: int = Field(
        DEFAULT_GENERATION_OPENAI_TIMEOUT,
        ge=1,
        description="Timeout in seconds for each OpenAI-compatible API request.",
    )
    stream: bool = Field(
        False,
        description="Use streaming OpenAI-compatible responses.",
    )
    extra_body: dict[str, Any] | None = Field(
        None,
        description="Additional JSON fields for OpenAI-compatible requests.",
    )
    api: str | None = Field(None, description="API to generate tests for.")
    docker_image: str | None = Field(
        None,
        description="Docker repository image (default: gso-<exp_id>:latest).",
    )
    docker_base_image: str | None = Field(
        DEFAULT_GENERATION_DOCKER_BASE_IMAGE,
        description="Base image used to build the Docker repository image.",
    )
    docker_repo_path: str | None = Field(
        None,
        description="Local repository checkout copied into the Docker image.",
    )
    docker_cache_dir: str | None = Field(
        None,
        description=(
            "Persistent per-repository package/compiler cache directory "
            "(default: <experiment>/docker_cache)."
        ),
    )
    docker_cpus: float | None = Field(None, gt=0, description="Docker CPU limit.")
    docker_memory: str | None = Field(None, description="Docker memory limit.")
    docker_platform: str | None = Field(
        None, description="Docker platform, for example linux/amd64."
    )
    rebuild_docker_image: bool = Field(
        False, description="Build the repository image without Docker layer cache."
    )
    keep_containers: bool = Field(
        False, description="Keep test-validation containers after execution."
    )
    keep_workspaces: bool = Field(
        False, description="Keep generated test-validation workspaces."
    )

    @classmethod
    def parse(cls, *args, **kwargs):
        if args and not kwargs.get("yaml_path"):
            kwargs["yaml_path"] = args[0]
        return cls(**kwargs)


class OversampleArgs(LLMArgs):
    exp_id: str | None = Field(None, description="Experiment ID.")
    model_name: str = Field("gpt-4o", description="Model name.")
    cache_batch_size: int = Field(100, description="Batch size for caching.")
    max_year: int = Field(2016, description="Maximum year for commits.")
    min_loc: int = Field(0, description="Minimum LOC for a commit.")
    multiprocess: int = Field(30, description="Num parallel processes for generation.")
    n: int = Field(5, ge=1, description="Num samples per generation.")
    max_tokens: int = Field(
        DEFAULT_GENERATION_MAX_TOKENS,
        ge=1,
        description="Maximum output tokens per generated performance test.",
    )
    openai_timeout: int = Field(
        DEFAULT_GENERATION_OPENAI_TIMEOUT,
        ge=1,
        description="Timeout in seconds for each OpenAI-compatible API request.",
    )
    api: str | None = Field(None, description="API to generate tests for.")

    @classmethod
    def parse(cls, *args, **kwargs):
        if args and not kwargs.get("yaml_path"):
            kwargs["yaml_path"] = args[0]
        return cls(**kwargs)
