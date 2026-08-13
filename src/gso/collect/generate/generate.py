import json
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import fire

from r2e.multiprocess import run_tasks_in_parallel
from gso.data import PerformanceCommit, Problem, Repo, Tests
from gso.logger import logger
from gso.constants import EXPS_DIR, ANALYSIS_APIS_DIR

from gso.utils.io import *
from gso.utils.llm import (
    configure_openai_compatible_llm,
    get_streaming_llm_completion,
)
from gso.collect.generate.prompt import *
from gso.collect.generate.helpers import *
from gso.collect.generate.context import prepare_mp_helper
from gso.collect.generate.args import PerfExpGenArgs

IS_RERUN_FLAG = False  # NOTE: runs testgen for valid probs from previous run
DEBUG_FLAG = False  # NOTE: debug flag to not overwrite existing tests
REASONING_MODELS = {"o1-mini", "o3-mini", "o1-preview", "o4-mini"}
SEMANTIC_RETRY_COUNT = 3


def create_generation_problem(
    repo: Repo, api: str, commits: list[PerformanceCommit], config: dict
) -> Problem:
    """Create a problem using configured install commands or model defaults."""
    return Problem.create_prob(repo, api, commits, config)


@dataclass(frozen=True)
class CommitGenerationTask:
    """All state required to prepare and generate tests for one commit."""

    problem_index: int
    commit_index: int
    repo: Repo
    problem: Problem
    commit: PerformanceCommit
    args: PerfExpGenArgs


@dataclass
class CommitGenerationResult:
    """Serializable result and diagnostics from one commit worker."""

    problem_index: int
    commit_index: int
    pid: str
    commit_hash: str
    commit_tests: Tests | None = None
    scenario_output: str | None = None
    scenarios: list[dict[str, str]] = field(default_factory=list)
    test_outputs: list[str] = field(default_factory=list)
    error: str | None = None

    def diagnostic(self) -> dict:
        return {
            "pid": self.pid,
            "commit_hash": self.commit_hash,
            "scenario_output": self.scenario_output,
            "scenarios": self.scenarios,
            "test_outputs": self.test_outputs,
            "error": self.error,
        }


def prepare_model_payload(
    model_name: str, messages: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Apply model-specific chat formatting without changing sampling behavior."""
    if model_name in REASONING_MODELS:
        prompt = "\n\n".join(message["content"] for message in messages)
        return [{"role": "user", "content": prompt}]
    return messages


def _prepare_semantic_retry(
    args: PerfExpGenArgs,
    payload: list[dict[str, str]],
    *,
    retry_number: int,
    previous_error: GeneratedTestError | None,
    output_requirement: str,
) -> tuple[PerfExpGenArgs, list[dict[str, str]]]:
    """Prepare a cache-bypassing request that corrects an invalid completion."""
    if retry_number == 0:
        return args, payload
    if previous_error is None:
        raise ValueError("A semantic retry requires the previous validation error")

    retry_payload = [dict(message) for message in payload]
    retry_payload.append(
        {
            "role": "user",
            "content": (
                "Automated validation rejected the previous completion:\n"
                f"{previous_error}\n\n"
                f"This is semantic retry {retry_number} of "
                f"{SEMANTIC_RETRY_COUNT}. Regenerate the complete response from "
                f"scratch. {output_requirement}"
            ),
        }
    )
    # A failed first attempt may already be cached. Retrying without cache ensures
    # that validation does not repeatedly receive the same invalid completion.
    retry_args = args.model_copy(update={"use_cache": False})
    return retry_args, retry_payload


def generate_commit_tests(task: CommitGenerationTask) -> CommitGenerationResult:
    """Generate one scenario plan and its tests sequentially for a commit."""
    result = CommitGenerationResult(
        problem_index=task.problem_index,
        commit_index=task.commit_index,
        pid=task.problem.pid,
        commit_hash=task.commit.commit_hash,
    )

    try:
        commit_tests = prepare_mp_helper((task.repo, task.problem, task.commit, False))
        context_message = commit_tests.chat_messages[1]
        scenario_task = SCENARIO_TASK_MSG.format(
            num_scenarios=task.args.n,
            api=task.problem.api,
            repo_name=task.repo.repo_name,
        )
        if task.repo.repo_instr:
            scenario_task += (
                "\n\nRepo-specific Instructions:\n" f"{task.repo.repo_instr}\n"
            )
        scenario_messages = [
            {"role": "system", "content": SCENARIO_SYSTEM_MSG},
            dict(context_message),
            {"role": "user", "content": scenario_task},
        ]
        scenario_payload = prepare_model_payload(
            task.args.model_name, scenario_messages
        )
        scenario_context = (
            f"{task.problem.pid}/{task.commit.quick_hash()} scenario plan"
        )
        scenario_error: GeneratedScenarioError | None = None
        for retry_number in range(SEMANTIC_RETRY_COUNT + 1):
            request_args, attempt_payload = _prepare_semantic_retry(
                task.args,
                scenario_payload,
                retry_number=retry_number,
                previous_error=scenario_error,
                output_requirement=(
                    "Return only one valid JSON object with every string value "
                    "double-quoted, and do not include Markdown or explanations."
                ),
            )
            result.scenario_output = get_streaming_llm_completion(
                request_args, attempt_payload
            )
            try:
                scenarios = get_generated_scenarios(
                    result.scenario_output,
                    expected_count=task.args.n,
                    context=scenario_context,
                )
            except GeneratedScenarioError as error:
                if retry_number == SEMANTIC_RETRY_COUNT:
                    error.add_note(
                        f"Failed after the initial request and "
                        f"{SEMANTIC_RETRY_COUNT} semantic retries."
                    )
                    raise
                logger.warning(
                    "%s failed validation; semantic retry %d/%d: %s",
                    scenario_context,
                    retry_number + 1,
                    SEMANTIC_RETRY_COUNT,
                    error,
                )
                scenario_error = error
                continue
            break
        result.scenarios = [scenario.model_dump() for scenario in scenarios]

        generated_tests = []
        for scenario_index, scenario in enumerate(scenarios, start=1):
            test_messages = [dict(message) for message in commit_tests.chat_messages]
            test_messages.append(
                {
                    "role": "user",
                    "content": SCENARIO_TEST_MSG.format(
                        scenario_number=scenario_index,
                        scenario_count=len(scenarios),
                        scenario_json=scenario.model_dump_json(indent=2),
                    ),
                }
            )
            test_payload = prepare_model_payload(task.args.model_name, test_messages)
            test_context = (
                f"{task.problem.pid}/{task.commit.quick_hash()} "
                f"scenario {scenario_index} test"
            )
            test_error: GeneratedTestError | None = None
            for retry_number in range(SEMANTIC_RETRY_COUNT + 1):
                request_args, attempt_payload = _prepare_semantic_retry(
                    task.args,
                    test_payload,
                    retry_number=retry_number,
                    previous_error=test_error,
                    output_requirement=(
                        "Return only one complete Python code block containing every "
                        "required function, with no explanations outside the block."
                    ),
                )
                raw_test = get_streaming_llm_completion(request_args, attempt_payload)
                result.test_outputs.append(raw_test)
                try:
                    generated_test = get_generated_test(
                        raw_test,
                        context=test_context,
                    )
                except GeneratedTestError as error:
                    if retry_number == SEMANTIC_RETRY_COUNT:
                        error.add_note(
                            f"Failed after the initial request and "
                            f"{SEMANTIC_RETRY_COUNT} semantic retries."
                        )
                        raise
                    logger.warning(
                        "%s failed validation; semantic retry %d/%d: %s",
                        test_context,
                        retry_number + 1,
                        SEMANTIC_RETRY_COUNT,
                        error,
                    )
                    test_error = error
                    continue
                generated_tests.append(generated_test)
                break

        if len(generated_tests) != task.args.n:
            raise GeneratedTestError(
                f"{task.problem.pid}/{task.commit.quick_hash()}: expected "
                f"{task.args.n} generated test(s), got {len(generated_tests)}"
            )
        commit_tests.add_samples(generated_tests)
        result.commit_tests = commit_tests
    except Exception:
        result.error = traceback.format_exc()

    return result


def save_problems_atomically(path: Path, problems: list[Problem]) -> None:
    """Merge generated problems and atomically replace the destination JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if path.exists():
            existing_data = load_json(path)
            if not isinstance(existing_data, list):
                raise ValueError(f"Existing problems file is not a JSON list: {path}")
            shutil.copyfile(path, temporary_path)
        save_problems(temporary_path, problems)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class PerfExpGenerator:
    """Generate performance testing problem/experiment for a repository's APIs"""

    def __init__(self, args):
        self.config = load_exp_config(args.yaml_path)
        self.configure_llm(args)
        self.exp_id = self.config["exp_id"]
        self.repo = Repo.from_url(self.config["repo_url"])

        # set repo-specific instructions
        if "repo_instr" in self.config:
            self.repo.repo_instr = self.config["repo_instr"]

        self.candidates = self.get_commit_map(self.repo)
        self.exp_dir = EXPS_DIR / self.exp_id

    def configure_llm(self, args) -> None:
        """Apply optional experiment LLM settings to test generation."""
        llm_config = self.config.get("llm")
        if llm_config is None:
            return
        if not isinstance(llm_config, dict):
            raise ValueError("The 'llm' experiment setting must be a YAML mapping")

        effective_llm_config = dict(llm_config)
        explicit_fields = getattr(args, "model_fields_set", set())
        if "model_name" in explicit_fields:
            effective_llm_config["model_name"] = args.model_name
        if "multiprocess" in explicit_fields:
            effective_llm_config["multiprocess"] = args.multiprocess
        if "max_tokens" in explicit_fields:
            effective_llm_config["max_tokens"] = args.max_tokens
        if "openai_timeout" in explicit_fields:
            effective_llm_config["openai_timeout"] = args.openai_timeout

        configured = configure_openai_compatible_llm(
            {"llm": effective_llm_config},
            default_model=args.model_name,
            default_multiprocess=args.multiprocess,
            default_max_tokens=args.max_tokens,
            default_openai_timeout=args.openai_timeout,
            model_env="GSO_GENERATION_MODEL",
            purpose="generation",
        )
        args.model_name = configured.model_name
        args.multiprocess = configured.multiprocess
        if configured.max_tokens is not None:
            args.max_tokens = configured.max_tokens
        if configured.openai_timeout is not None:
            args.openai_timeout = configured.openai_timeout

    def get_commit_map(self, repo: Repo):
        """Get the api-commit map for the repository"""
        ac_map = load_map(ANALYSIS_APIS_DIR / f"{repo.repo_name}_ac_map.json")
        return ac_map.api_to_commits

    def save_failed_completions(self, outputs, args, error: Exception) -> Path | None:
        """Keep invalid raw output for diagnosis without touching the problems file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.exp_dir / f"{self.exp_id}_generation_failed_{timestamp}.json"
        try:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            with path.open("w") as file:
                json.dump(
                    {
                        "model_name": args.model_name,
                        "max_tokens": args.max_tokens,
                        "tests_per_commit": args.n,
                        "choices_per_request": 1,
                        "error": str(error),
                        "outputs": outputs,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
            return path
        except Exception as save_error:
            logger.error(f"Failed to save invalid completions: {save_error}")
            return None

    def gen(self, args) -> list[Problem]:
        logger.debug(f"Generating perftests: {self.repo}")

        problems = [
            create_generation_problem(self.repo, api, commits, self.config)
            for api, commits in self.candidates.items()
        ]

        if args.api:
            problems = [problem for problem in problems if problem.api == args.api]
            if not problems:
                raise ValueError(f"No problem found for API: {args.api}")

        if IS_RERUN_FLAG:
            prev_run = load_problems(self.exp_dir / f"{self.exp_id}_results.json")
            prev_valid_apis = [
                problem.api for problem in prev_run if problem.is_valid()
            ]
            problems = [
                problem for problem in problems if problem.api in prev_valid_apis
            ]

        for problem in problems:
            problem.filter_commits_year(args.max_year)
            problem.filter_commits_loc(args.min_loc)
        problems = [problem for problem in problems if problem.num_commits() > 0]
        if not problems:
            raise ValueError("No candidate commits remain after generation filters")

        commit_tasks = [
            CommitGenerationTask(
                problem_index=problem_index,
                commit_index=commit_index,
                repo=self.repo,
                problem=problem,
                commit=commit,
                args=args,
            )
            for problem_index, problem in enumerate(problems)
            for commit_index, commit in enumerate(problem.commits)
        ]
        if not commit_tasks:
            raise RuntimeError("No commit generation tasks were prepared")

        commit_workers = min(args.multiprocess, len(commit_tasks))
        print(
            f"Generating {len(problems)} problems across {len(commit_tasks)} commits "
            f"(tests_per_commit={args.n}, choices_per_request=1, "
            f"commit_workers={commit_workers}, max_tokens={args.max_tokens}, "
            "stream=True)"
        )
        generation_outputs = run_tasks_in_parallel(
            generate_commit_tests,
            commit_tasks,
            use_progress_bar=True,
            num_workers=commit_workers,
            progress_bar_desc="Generating commit tests",
        )

        generation_errors = []
        failure_diagnostics = []
        generated_by_commit: dict[tuple[int, int], Tests] = {}
        if len(generation_outputs) != len(commit_tasks):
            generation_errors.append(
                f"expected {len(commit_tasks)} commit result(s), "
                f"got {len(generation_outputs)}"
            )

        for task, output in zip(commit_tasks, generation_outputs):
            task_label = f"{task.problem.pid}/{task.commit.quick_hash()}"
            if not output.is_success():
                error = output.exception_tb or f"worker status: {output.status}"
                generation_errors.append(f"{task_label}: {error}")
                failure_diagnostics.append(
                    {
                        "pid": task.problem.pid,
                        "commit_hash": task.commit.commit_hash,
                        "error": error,
                    }
                )
                continue

            commit_result = output.result
            if not isinstance(commit_result, CommitGenerationResult):
                error = f"worker returned {type(commit_result).__name__}"
                generation_errors.append(f"{task_label}: {error}")
                failure_diagnostics.append(
                    {
                        "pid": task.problem.pid,
                        "commit_hash": task.commit.commit_hash,
                        "error": error,
                    }
                )
                continue
            if commit_result.error is not None:
                generation_errors.append(f"{task_label}: {commit_result.error}")
                failure_diagnostics.append(commit_result.diagnostic())
                continue
            if commit_result.commit_tests is None:
                error = "worker returned no commit tests"
                generation_errors.append(f"{task_label}: {error}")
                failure_diagnostics.append(commit_result.diagnostic())
                continue
            if commit_result.commit_tests.num_samples() != args.n:
                error = (
                    f"expected {args.n} test sample(s), got "
                    f"{commit_result.commit_tests.num_samples()}"
                )
                generation_errors.append(f"{task_label}: {error}")
                failure_diagnostics.append(commit_result.diagnostic())
                continue

            expected_key = (task.problem_index, task.commit_index)
            key = (commit_result.problem_index, commit_result.commit_index)
            if key != expected_key:
                error = f"expected commit result key {expected_key}, got {key}"
                generation_errors.append(f"{task_label}: {error}")
                failure_diagnostics.append(commit_result.diagnostic())
                continue
            if key in generated_by_commit:
                error = f"duplicate commit result key: {key}"
                generation_errors.append(f"{task_label}: {error}")
                failure_diagnostics.append(commit_result.diagnostic())
                continue
            generated_by_commit[key] = commit_result.commit_tests

        if generation_errors:
            displayed_errors = generation_errors[:20]
            details = "\n".join(f"- {error}" for error in displayed_errors)
            if len(generation_errors) > len(displayed_errors):
                details += (
                    f"\n- ... and {len(generation_errors) - len(displayed_errors)} "
                    "more error(s)"
                )
            error = GeneratedTestError(
                "Failed to generate all commit scenarios/tests:\n" + details
            )
            failed_path = self.save_failed_completions(failure_diagnostics, args, error)
            artifact_message = (
                f" Invalid completions were saved to {failed_path}."
                if failed_path is not None
                else ""
            )
            raise GeneratedTestError(
                f"{error}\nExisting problems file was not modified.{artifact_message}"
            ) from error

        for problem_index, problem in enumerate(problems):
            problem.set_tests(
                [
                    generated_by_commit[(problem_index, commit_index)]
                    for commit_index in range(problem.num_commits())
                ]
            )
        validate_problem_test_samples(problems)

        results_json = f"{self.exp_id}_problems{'_DEBUG' if DEBUG_FLAG else ''}.json"
        results_path = self.exp_dir / results_json
        save_problems_atomically(results_path, problems)
        print(f"Saved validated generated problems to {results_path}")
        return problems


if __name__ == "__main__":
    args = fire.Fire(PerfExpGenArgs.parse)
    generator = PerfExpGenerator(args)
    generator.gen(args)
