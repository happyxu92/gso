import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from concurrent.futures import as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import fire
from pebble import ProcessPool

from r2e.multiprocess import TaskResult, TaskRunStatus
from gso.data import PerformanceCommit, Problem, Repo, Tests
from gso.logger import logger
from gso.constants import (
    ANALYSIS_APIS_DIR,
    ANALYSIS_REPOS_DIR,
    EXPS_DIR,
    LLM_CACHE_STAGES,
)
from gso.collect.execute.execute import (
    GeneratedTestExecutionConfig,
    evaluate_generated_test,
    prepare_generated_test_execution,
)

from gso.utils.io import *
from gso.utils.llm import (
    configure_openai_compatible_llm,
    get_llm_completion,
)
from gso.collect.generate.prompt import *
from gso.collect.generate.helpers import *
from gso.collect.generate.context import prepare_mp_helper
from gso.collect.generate.args import PerfExpGenArgs

IS_RERUN_FLAG = False  # NOTE: runs testgen for valid probs from previous run
DEBUG_FLAG = False  # NOTE: debug flag to not overwrite existing tests
REASONING_MODELS = {"o1-mini", "o3-mini", "o1-preview", "o4-mini"}
SEMANTIC_RETRY_COUNT = 3
DEFAULT_GENERATION_LLM_CACHE_SETTINGS = {
    "test_generation": False,
}
GENERATION_PROGRESS_PREFIX = "Generation progress:"
GENERATION_EVENT_PREFIX = "Generation event:"


def log_generation_event(test_context: str, event: str) -> None:
    """Emit a structured lifecycle event that run_pipeline.py can relay."""
    print(f"{GENERATION_EVENT_PREFIX} {test_context} {event}", flush=True)


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
    execution_config: GeneratedTestExecutionConfig | None = None
    target_index: int = 0
    target_count: int = 1


@dataclass
class CommitGenerationResult:
    """Serializable result and diagnostics from one commit worker."""

    problem_index: int
    commit_index: int
    pid: str
    commit_hash: str
    commit_tests: Tests | None = None
    scenarios: list[dict[str, str]] = field(default_factory=list)
    test_outputs: list[str] = field(default_factory=list)
    test_attempts: list[dict] = field(default_factory=list)
    failed_test_slots: list[dict] = field(default_factory=list)
    error: str | None = None

    def diagnostic(self) -> dict:
        return {
            "pid": self.pid,
            "commit_hash": self.commit_hash,
            "scenarios": self.scenarios,
            "test_outputs": self.test_outputs,
            "test_attempts": self.test_attempts,
            "failed_test_slots": self.failed_test_slots,
            "error": self.error,
        }


FAILED_SCENARIO_ERROR_MAX_CHARS = 800


def _summarize_failed_scenario(
    scenario: TestScenario | None, error: GeneratedTestError
) -> dict:
    """Compact ``{title, error}`` record for a failed scenario slot.

    Full per-attempt diagnostics (raw output, execution log, full error) are kept in
    ``CommitGenerationResult``; this summary only steers the next scenario away from
    repeating a workload whose test was already rejected.
    """
    title = scenario.title if scenario is not None else "<unparsed scenario>"
    message = str(error)
    if len(message) > FAILED_SCENARIO_ERROR_MAX_CHARS:
        message = message[:FAILED_SCENARIO_ERROR_MAX_CHARS] + " ...(truncated)"
    return {"title": title, "error": message}


def format_previous_scenarios(
    scenarios: list[TestScenario],
    failed_scenarios: list[dict] | None = None,
) -> str:
    """Format successful and failed scenarios for the next LLM request.

    Successful scenarios are listed in full so the model can produce materially
    different workloads. Failed scenarios are listed as ``{title, error}`` only, so
    the model avoids reusing a scenario whose test was already rejected.
    """
    has_successful = bool(scenarios)
    has_failed = bool(failed_scenarios)
    if not has_successful and not has_failed:
        return "None. This is the first scenario."

    sections: list[str] = []
    if has_successful:
        sections.append(
            "Successful scenarios:\n"
            + json.dumps(
                [scenario.model_dump() for scenario in scenarios],
                indent=2,
                ensure_ascii=False,
            )
        )
    if has_failed:
        sections.append(
            "Failed scenarios (do not reuse these; the rejection reason is shown):\n"
            + json.dumps(failed_scenarios, indent=2, ensure_ascii=False)
        )
    return "\n\n".join(sections)


def add_scenario_comment(test: str, scenario: TestScenario) -> str:
    """Persist the scenario alongside its generated test as Python comments."""
    lines = ["# GSO generated scenario:"]
    for name, value in scenario.model_dump().items():
        normalized = " ".join(value.split())
        lines.append(f"# {name}: {normalized}")
    return "\n".join(lines) + "\n\n" + test.lstrip()


def prepare_model_payload(
    model_name: str, messages: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Merge user content and apply model-specific chat formatting."""
    user_content = "\n\n".join(
        message["content"] for message in messages if message["role"] == "user"
    )
    merged_messages: list[dict[str, str]] = []
    user_added = False
    for message in messages:
        if message["role"] != "user":
            merged_messages.append(dict(message))
        elif not user_added:
            merged_messages.append({"role": "user", "content": user_content})
            user_added = True

    if model_name in REASONING_MODELS:
        prompt = "\n\n".join(message["content"] for message in merged_messages)
        return [{"role": "user", "content": prompt}]
    return merged_messages


def _prepare_generation_messages(
    base_messages: list[dict[str, str]], task_message: str
) -> list[dict[str, str]]:
    """Replace the generic prepared task with the scenario-specific task.

    ``prepare_mp_helper`` supplies separate user messages for commit context and a
    legacy "write a test" task. The detailed scenario task supersedes that generic
    task, so retaining both only repeats instructions (including repo guidance).
    """
    messages = [dict(message) for message in base_messages]
    user_indices = [
        index for index, message in enumerate(messages) if message["role"] == "user"
    ]
    if len(user_indices) >= 2:
        messages.pop(user_indices[-1])
    messages.append({"role": "user", "content": task_message})
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

    retry_instruction = (
        "Automated validation rejected the previous completion:\n"
        f"{previous_error}\n\n"
        f"This is semantic retry {retry_number} of "
        f"{SEMANTIC_RETRY_COUNT}. Regenerate the complete response from "
        f"scratch. {output_requirement}"
    )
    retry_payload = [dict(message) for message in payload]
    user_message = next(
        (message for message in retry_payload if message["role"] == "user"), None
    )
    if user_message is None:
        retry_payload.append({"role": "user", "content": retry_instruction})
    else:
        user_message["content"] += f"\n\n{retry_instruction}"
    # A failed first attempt may already be cached. Retrying without cache ensures
    # that validation does not repeatedly receive the same invalid completion.
    retry_args = args.model_copy(update={"use_cache": False})
    return retry_args, retry_payload


def generate_commit_tests(task: CommitGenerationTask) -> CommitGenerationResult:
    """Generate, execute, and accept one scenario/test at a time for a commit."""
    generation_started = time.monotonic()
    llm_seconds = 0.0
    validation_seconds = 0.0
    result = CommitGenerationResult(
        problem_index=task.problem_index,
        commit_index=task.commit_index,
        pid=task.problem.pid,
        commit_hash=task.commit.commit_hash,
    )

    try:
        commit_tests = prepare_mp_helper((task.repo, task.problem, task.commit, False))
        generated_tests = []
        successful_scenarios: list[TestScenario] = []
        failed_scenarios: list[dict] = []
        for scenario_index in range(1, task.args.n + 1):
            slot_status = "skipped"
            combined_task = SCENARIO_TEST_MSG.format(
                api=task.problem.api,
                repo_name=task.repo.repo_name,
                scenario_number=scenario_index,
                scenario_count=task.args.n,
                previous_scenarios=format_previous_scenarios(
                    successful_scenarios, failed_scenarios
                ),
            )
            if task.repo.repo_instr:
                combined_task += (
                    "\n\nRepo-specific Instructions:\n" f"{task.repo.repo_instr}\n"
                )
            combined_messages = _prepare_generation_messages(
                commit_tests.chat_messages, combined_task
            )
            combined_payload = prepare_model_payload(
                task.args.model_name, combined_messages
            )
            test_context = (
                f"{task.problem.pid}/{task.commit.quick_hash()} "
                f"scenario/test {scenario_index}"
            )
            event_context = (
                f"target {task.target_index + 1}/{task.target_count} | "
                f"{task.problem.pid}/{task.commit.quick_hash()} | "
                f"test {scenario_index}/{task.args.n} |"
            )
            test_error: GeneratedTestError | None = None
            for retry_number in range(SEMANTIC_RETRY_COUNT + 1):
                execution_log = None
                scenario = None
                request_args, attempt_payload = _prepare_semantic_retry(
                    task.args,
                    combined_payload,
                    retry_number=retry_number,
                    previous_error=test_error,
                    output_requirement=(
                        "Return the complete response from scratch: exactly one JSON "
                        "scenario block followed by exactly one complete Python test "
                        "block, with no explanations outside the blocks. The scenario "
                        "may change from the rejected attempt."
                    ),
                )
                raw_output = None
                try:
                    if retry_number == 0:
                        log_generation_event(event_context, "requesting test")
                    else:
                        log_generation_event(
                            event_context,
                            f"re-requesting test (semantic retry "
                            f"{retry_number}/{SEMANTIC_RETRY_COUNT})",
                        )
                    try:
                        completion_started = time.monotonic()
                        try:
                            raw_output = get_llm_completion(
                                request_args,
                                attempt_payload,
                                stream=request_args.stream,
                            )
                        finally:
                            llm_seconds += time.monotonic() - completion_started
                    except Exception as completion_exception:
                        raise GeneratedTestError(
                            f"{test_context}: LLM completion raised "
                            f"{type(completion_exception).__name__}: "
                            f"{completion_exception}"
                        ) from completion_exception
                    result.test_outputs.append(raw_output)
                    scenario, generated_test = get_generated_scenario_and_test(
                        raw_output, context=test_context
                    )
                    generated_test = add_scenario_comment(
                        generated_test,
                        scenario,
                    )
                    if task.execution_config is not None:
                        log_generation_event(
                            event_context,
                            f"starting test (attempt {retry_number + 1}/"
                            f"{SEMANTIC_RETRY_COUNT + 1})",
                        )
                        try:
                            validation_started = time.monotonic()
                            try:
                                execution_result = evaluate_generated_test(
                                    task.problem,
                                    task.commit,
                                    generated_test,
                                    task.execution_config,
                                )
                            finally:
                                validation_seconds += (
                                    time.monotonic() - validation_started
                                )
                        except Exception as execution_exception:
                            log_generation_event(
                                event_context,
                                "test failed "
                                f"({type(execution_exception).__name__}: "
                                f"{execution_exception})",
                            )
                            raise GeneratedTestError(
                                f"{test_context}: automated execution raised "
                                f"{type(execution_exception).__name__}: "
                                f"{execution_exception}"
                            ) from execution_exception
                        execution_log = execution_result.log_path
                        if not execution_result.passed:
                            execution_error = execution_result.error or (
                                "Generated test execution failed without diagnostics"
                            )
                            log_generation_event(
                                event_context,
                                "test failed"
                                + (
                                    f" (log: {execution_log})"
                                    if execution_log is not None
                                    else ""
                                ),
                            )
                            raise GeneratedTestError(
                                f"{test_context}: automated execution failed:\n"
                                f"{execution_error}"
                            )
                        log_generation_event(event_context, "test passed")
                except GeneratedTestError as error:
                    attempt = {
                        "scenario_index": scenario_index,
                        "retry_number": retry_number,
                        "scenario": scenario.model_dump() if scenario else None,
                        "raw_output": raw_output,
                        "error": str(error),
                        "execution_log": execution_log,
                    }
                    result.test_attempts.append(attempt)
                    if retry_number == SEMANTIC_RETRY_COUNT:
                        result.failed_test_slots.append(
                            {
                                "scenario_index": scenario_index,
                                "error": str(error),
                                "execution_log": execution_log,
                            }
                        )
                        failed_scenarios.append(
                            _summarize_failed_scenario(scenario, error)
                        )
                        logger.error(
                            "%s failed after the initial request and %d semantic "
                            "retries; skipping this test slot",
                            test_context,
                            SEMANTIC_RETRY_COUNT,
                        )
                        break
                    logger.warning(
                        "%s failed validation; semantic retry %d/%d: %s",
                        test_context,
                        retry_number + 1,
                        SEMANTIC_RETRY_COUNT,
                        error,
                    )
                    test_error = error
                    continue
                result.test_attempts.append(
                    {
                        "scenario_index": scenario_index,
                        "retry_number": retry_number,
                        "scenario": scenario.model_dump(),
                        "raw_output": raw_output,
                        "error": None,
                        "execution_log": execution_log,
                    }
                )
                generated_tests.append(generated_test)
                successful_scenarios.append(scenario)
                slot_status = "accepted"
                break

            semantic_retries = sum(
                attempt["retry_number"] > 0 for attempt in result.test_attempts
            )
            slot_completion = (
                " | slots complete" if scenario_index == task.args.n else ""
            )
            elapsed_seconds = time.monotonic() - generation_started
            print(
                f"{GENERATION_PROGRESS_PREFIX} "
                f"target {task.target_index + 1}/{task.target_count} | "
                f"{task.problem.pid}/{task.commit.quick_hash()} | "
                f"test {scenario_index}/{task.args.n} {slot_status} | "
                f"accepted={len(generated_tests)} | "
                f"skipped={len(result.failed_test_slots)} | "
                f"semantic_retries={semantic_retries} | "
                f"elapsed={elapsed_seconds:.1f}s | "
                f"llm={llm_seconds:.1f}s | "
                f"validation={validation_seconds:.1f}s{slot_completion}",
                flush=True,
            )

        result.scenarios = [scenario.model_dump() for scenario in successful_scenarios]

        if not generated_tests:
            raise GeneratedTestError(
                f"{task.problem.pid}/{task.commit.quick_hash()}: no valid tests were "
                f"generated after attempting {task.args.n} test slot(s)"
            )
        commit_tests.add_samples(generated_tests)
        result.commit_tests = commit_tests
    except Exception:
        result.error = traceback.format_exc()

    return result


def generate_commits_as_completed(tasks: list[CommitGenerationTask], num_workers: int):
    """Yield each task result as soon as its worker finishes."""
    with ProcessPool(
        max_workers=num_workers,
        context=mp.get_context("spawn"),
    ) as pool:
        future_tasks = {
            pool.schedule(generate_commit_tests, args=[task]): task for task in tasks
        }
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                result = future.result()
            except Exception:
                yield task, TaskResult(
                    status=TaskRunStatus.EXCEPTION,
                    exception_tb=traceback.format_exc(),
                )
            else:
                yield task, TaskResult(
                    status=TaskRunStatus.SUCCESS,
                    result=result,
                )


def load_existing_problems(path: Path) -> list[Problem]:
    """Load a reusable problems snapshot, rejecting an invalid top-level shape."""
    if not path.exists():
        return []

    existing_data = load_json(path)
    if not isinstance(existing_data, list):
        raise ValueError(f"Existing problems file is not a JSON list: {path}")
    return [Problem(**problem) for problem in existing_data]


def merge_generated_problems(
    existing_problems: list[Problem], new_problems: list[Problem]
) -> list[Problem]:
    """Merge problem snapshots without replacing existing API/commit results."""
    merged_problems = [problem.model_copy(deep=True) for problem in existing_problems]
    problem_indexes = {
        problem.api: problem_index
        for problem_index, problem in enumerate(merged_problems)
    }

    for new_problem in new_problems:
        problem_index = problem_indexes.get(new_problem.api)
        if problem_index is None:
            problem_indexes[new_problem.api] = len(merged_problems)
            merged_problems.append(new_problem.model_copy(deep=True))
            continue

        merged_problem = merged_problems[problem_index]
        existing_commit_hashes = {
            commit.commit_hash for commit in merged_problem.commits
        }
        merged_problem.commits.extend(
            commit.model_copy(deep=True)
            for commit in new_problem.commits
            if commit.commit_hash not in existing_commit_hashes
        )

        existing_test_hashes = {tests.commit_hash for tests in merged_problem.tests}
        merged_problem.tests.extend(
            tests.model_copy(deep=True)
            for tests in new_problem.tests
            if tests.commit_hash not in existing_test_hashes
        )

    return merged_problems


def save_problems_atomically(path: Path, problems: list[Problem]) -> None:
    """Merge API/commit results and atomically replace the destination JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        existing_problems = load_existing_problems(path)
        merged_problems = merge_generated_problems(existing_problems, problems)
        save_problems(temporary_path, merged_problems)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_generated_problems(
    problems: list[Problem],
    generated_by_commit: dict[tuple[int, int], Tests],
) -> list[Problem]:
    """Build a persistable snapshot from the commit results received so far."""
    generated_problems = []
    for problem_index, source_problem in enumerate(problems):
        successful_commit_entries = [
            (commit_index, generated_by_commit[(problem_index, commit_index)])
            for commit_index in range(source_problem.num_commits())
            if (problem_index, commit_index) in generated_by_commit
        ]
        if not successful_commit_entries:
            continue

        problem = source_problem.model_copy(deep=True)
        problem.commits = [
            source_problem.commits[commit_index]
            for commit_index, _ in successful_commit_entries
        ]
        problem.set_tests(
            [commit_tests for _, commit_tests in successful_commit_entries]
        )
        generated_problems.append(problem)
    return generated_problems


class PerfExpGenerator:
    """Generate performance testing problem/experiment for a repository's APIs"""

    def __init__(self, args):
        self.config = load_exp_config(args.yaml_path)
        self.llm_cache_settings = DEFAULT_GENERATION_LLM_CACHE_SETTINGS.copy()
        self.configure_llm(args)
        self.exp_id = self.config["exp_id"]
        self.repo = Repo.from_url(self.config["repo_url"])

        # set repo-specific instructions
        if "repo_instr" in self.config:
            self.repo.repo_instr = self.config["repo_instr"]

        self.candidates = self.get_commit_map(self.repo)
        self.exp_dir = EXPS_DIR / self.exp_id

    def configure_llm(self, args) -> None:
        """Apply optional experiment LLM settings to test generation.

        Per-stage caching is read from ``llm.cache`` (mirroring ``commits.py``).
        Generation defaults to ``False`` so repeated runs hit the live endpoint
        instead of reusing stale disk-cache entries; an explicit ``--use_cache``
        CLI flag always takes precedence over the YAML setting.
        """
        raw_llm_config = self.config.get("llm")
        llm_config = raw_llm_config if raw_llm_config is not None else {}
        if not isinstance(llm_config, dict):
            raise ValueError("The 'llm' experiment setting must be a YAML mapping")

        cache_config = llm_config.get("cache", {})
        if cache_config is None:
            cache_config = {}
        if not isinstance(cache_config, dict):
            raise ValueError("llm.cache must be a YAML mapping")
        unknown_stages = cache_config.keys() - LLM_CACHE_STAGES
        if unknown_stages:
            stages = ", ".join(sorted(unknown_stages))
            raise ValueError(f"Unsupported llm.cache stage(s): {stages}")
        for stage, value in cache_config.items():
            if not isinstance(value, bool):
                raise ValueError(f"llm.cache.{stage} must be a boolean")
        self.llm_cache_settings = {
            stage: cache_config.get(stage, default)
            for stage, default in DEFAULT_GENERATION_LLM_CACHE_SETTINGS.items()
        }

        explicit_fields = getattr(args, "model_fields_set", set())
        if "use_cache" in explicit_fields:
            self.llm_cache_settings = {
                **self.llm_cache_settings,
                "test_generation": args.use_cache,
            }
        else:
            args.use_cache = self.llm_cache_settings["test_generation"]

        if raw_llm_config is None:
            return

        effective_llm_config = dict(llm_config)
        if "model_name" in explicit_fields:
            effective_llm_config["model_name"] = args.model_name
        if "multiprocess" in explicit_fields:
            effective_llm_config["multiprocess"] = args.multiprocess
        if "max_tokens" in explicit_fields:
            effective_llm_config["max_tokens"] = args.max_tokens
        if "openai_timeout" in explicit_fields:
            effective_llm_config["openai_timeout"] = args.openai_timeout
        if "stream" in explicit_fields:
            effective_llm_config["stream"] = args.stream
        if "extra_body" in explicit_fields:
            effective_llm_config["extra_body"] = args.extra_body

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
        args.stream = configured.stream
        args.extra_body = configured.extra_body

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

    def save_retry_diagnostics(self, outputs, args) -> Path | None:
        """Persist retry, skipped-slot, and skipped-commit diagnostics."""
        if not outputs:
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.exp_dir / f"{self.exp_id}_generation_retries_{timestamp}.json"
        try:
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            with path.open("w") as file:
                json.dump(
                    {
                        "model_name": args.model_name,
                        "max_tokens": args.max_tokens,
                        "tests_per_commit": args.n,
                        "outputs": outputs,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
            return path
        except Exception as save_error:
            logger.error(f"Failed to save retry diagnostics: {save_error}")
            return None

    def gen(self, args) -> list[Problem]:
        logger.debug(f"Generating perftests: {self.repo}")

        results_json = f"{self.exp_id}_problems{'_DEBUG' if DEBUG_FLAG else ''}.json"
        results_path = self.exp_dir / results_json

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

        existing_problems = load_existing_problems(results_path)
        existing_api_commits = {
            (problem.api, commit.commit_hash)
            for problem in existing_problems
            for commit in problem.commits
        }
        skipped_commit_count = 0
        for problem in problems:
            candidate_count = problem.num_commits()
            problem.commits = [
                commit
                for commit in problem.commits
                if (problem.api, commit.commit_hash) not in existing_api_commits
            ]
            skipped_commit_count += candidate_count - problem.num_commits()
        problems = [problem for problem in problems if problem.num_commits() > 0]

        if skipped_commit_count:
            print(
                f"Reusing {skipped_commit_count} existing API/commit result(s) "
                f"from {results_path}",
                flush=True,
            )
        if not problems:
            print("All candidate API/commit results already exist; nothing to generate")
            return existing_problems

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
        commit_tasks = [
            replace(
                task,
                target_index=target_index,
                target_count=len(commit_tasks),
            )
            for target_index, task in enumerate(commit_tasks)
        ]

        repo_path = args.docker_repo_path
        if repo_path is None:
            analyzed_repo = ANALYSIS_REPOS_DIR / self.repo.repo_name
            repo_path = str(analyzed_repo) if analyzed_repo.is_dir() else None
        execution_config = prepare_generated_test_execution(
            problems,
            GeneratedTestExecutionConfig(
                exp_id=self.exp_id,
                backend="docker",
                docker_image=args.docker_image,
                docker_cpus=args.docker_cpus,
                docker_memory=args.docker_memory,
                docker_platform=args.docker_platform,
                docker_base_image=args.docker_base_image,
                docker_repo_path=repo_path,
                rebuild_docker_image=args.rebuild_docker_image,
                keep_containers=args.keep_containers,
                keep_workspaces=args.keep_workspaces,
            ),
        )
        commit_tasks = [
            replace(task, execution_config=execution_config) for task in commit_tasks
        ]

        commit_workers = min(args.multiprocess, len(commit_tasks))
        print(
            f"Generating {len(problems)} problems across {len(commit_tasks)} commits "
            f"(tests_per_commit={args.n}, choices_per_request=1, "
            f"commit_workers={commit_workers}, max_tokens={args.max_tokens}, "
            f"stream={args.stream})"
        )
        generation_outputs = generate_commits_as_completed(commit_tasks, commit_workers)

        generation_errors = []
        failure_diagnostics = []
        retry_diagnostics = []
        generated_by_commit: dict[tuple[int, int], Tests] = {}
        received_output_count = 0
        for task, output in generation_outputs:
            received_output_count += 1
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
            if any(
                attempt.get("error") is not None
                for attempt in commit_result.test_attempts
            ):
                retry_diagnostics.append(commit_result.diagnostic())
            if commit_result.commit_tests is None:
                error = "worker returned no commit tests"
                generation_errors.append(f"{task_label}: {error}")
                failure_diagnostics.append(commit_result.diagnostic())
                continue
            sample_count = commit_result.commit_tests.num_samples()
            if not 1 <= sample_count <= args.n:
                error = f"expected 1-{args.n} test sample(s), got {sample_count}"
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

            # Persist every completed API/commit immediately. The snapshot contains
            # all successful commits received so far, while the atomic replacement
            # keeps the existing JSON file valid if generation is interrupted.
            generated_problems = build_generated_problems(problems, generated_by_commit)
            validate_problem_test_samples(generated_problems)
            save_problems_atomically(results_path, generated_problems)
            print(
                f"Saved completed API/commit {task_label} to {results_path} "
                f"({len(generated_by_commit)}/{len(commit_tasks)})",
                flush=True,
            )

        if received_output_count != len(commit_tasks):
            generation_errors.append(
                f"expected {len(commit_tasks)} commit result(s), "
                f"got {received_output_count}"
            )

        generated_problems = build_generated_problems(problems, generated_by_commit)
        for problem_index, problem in enumerate(problems):
            if not any(key[0] == problem_index for key in generated_by_commit):
                logger.error(
                    "%s: no commits produced valid generated tests; skipping problem",
                    problem.pid,
                )

        if not generated_problems:
            displayed_errors = generation_errors[:20]
            details = "\n".join(f"- {error}" for error in displayed_errors)
            if len(generation_errors) > len(displayed_errors):
                details += (
                    f"\n- ... and {len(generation_errors) - len(displayed_errors)} "
                    "more error(s)"
                )
            error = GeneratedTestError(
                "No commits produced valid generated tests"
                + (f":\n{details}" if details else "")
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

        if generation_errors:
            logger.warning(
                "Skipping %d generation issue(s); continuing with %d successful "
                "commit(s)",
                len(generation_errors),
                len(generated_by_commit),
            )

        validate_problem_test_samples(generated_problems)
        print(f"Saved validated generated problems to {results_path}")
        diagnostic_outputs = [*retry_diagnostics, *failure_diagnostics]
        retry_path = self.save_retry_diagnostics(diagnostic_outputs, args)
        if retry_path is not None:
            print(f"Saved generation retry/failure diagnostics to {retry_path}")
        return merge_generated_problems(existing_problems, generated_problems)


if __name__ == "__main__":
    args = fire.Fire(PerfExpGenArgs.parse)
    generator = PerfExpGenerator(args)
    generator.gen(args)
