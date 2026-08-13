from r2e.llms.llm_args import LLMArgs
from pydantic import Field


DEFAULT_GENERATION_MAX_TOKENS = 32768
DEFAULT_GENERATION_OPENAI_TIMEOUT = 600


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
            "Number of scenarios/tests generated per commit; each LLM request "
            "returns one choice."
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
    api: str | None = Field(None, description="API to generate tests for.")

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
