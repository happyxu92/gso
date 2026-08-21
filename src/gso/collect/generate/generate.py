import atexit
import json
from contextlib import nullcontext
import multiprocessing as mp
import os
import re
import stat
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import fire
from pebble import ProcessPool
import yaml

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
    GeneratedTestExecutionResult,
    cleanup_prepared_test_execution,
    evaluate_generated_test,
    open_generated_test_validation_session,
    prepare_commit_test_execution,
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
from gso.collect.generate.api_context import build_parent_api_preflight_test
from gso.collect.generate.args import PerfExpGenArgs

IS_RERUN_FLAG = False  # NOTE: runs testgen for valid probs from previous run
DEBUG_FLAG = False  # NOTE: debug flag to not overwrite existing tests
REASONING_MODELS = {"o1-mini", "o3-mini", "o1-preview", "o4-mini"}
SEMANTIC_RETRY_COUNT = 3
MAX_COMMIT_PREPARATION_WORKERS = 4
DEFAULT_GENERATION_LLM_CACHE_SETTINGS = {
    "test_generation": False,
}
GENERATION_PROGRESS_PREFIX = "Generation progress:"
GENERATION_EVENT_PREFIX = "Generation event:"
API_PREFLIGHT_RESULT_PREFIX = "GSO_API_PREFLIGHT_RESULTS="
CODEX_INSTALL_COMMAND_TIMEOUT_SECONDS = 600
_PREPARED_EXECUTION_CONFIGS: dict[str, GeneratedTestExecutionConfig] = {}
_PREPARED_EXECUTION_CONFIGS_LOCK = threading.Lock()


CODEX_INSTALL_COMMAND_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "install_commands": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        }
    },
    "required": ["install_commands"],
    "additionalProperties": False,
}


def _register_prepared_execution(config: GeneratedTestExecutionConfig) -> None:
    if config.docker_image is None or config.prepared_commit_hash is None:
        return
    with _PREPARED_EXECUTION_CONFIGS_LOCK:
        _PREPARED_EXECUTION_CONFIGS[config.docker_image] = config


def _cleanup_prepared_execution(config: GeneratedTestExecutionConfig) -> None:
    image = config.docker_image
    if image is None:
        return
    try:
        cleanup_prepared_test_execution(config)
    except Exception as error:
        logger.warning("Failed to remove prepared image %s: %s", image, error)
        return
    with _PREPARED_EXECUTION_CONFIGS_LOCK:
        _PREPARED_EXECUTION_CONFIGS.pop(image, None)


def _cleanup_all_prepared_executions() -> None:
    with _PREPARED_EXECUTION_CONFIGS_LOCK:
        configs = list(_PREPARED_EXECUTION_CONFIGS.values())
    for config in configs:
        _cleanup_prepared_execution(config)


atexit.register(_cleanup_all_prepared_executions)


def request_install_commands_from_codex(
    repo_path: str | Path,
    py_version: str,
) -> list[str] | None:
    """Ask Codex to inspect a local repository and propose install commands.

    Codex runs with a read-only sandbox and a response schema. Failures are
    intentionally non-fatal so generation can fall back to ``Problem`` defaults.
    """
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.is_dir():
        logger.warning(
            f"Cannot infer install commands: repository path does not exist: "
            f"{repo_path}"
        )
        return None

    prompt = f"""Inspect the repository in your current working directory and determine
the shell commands needed to install it for GSO performance-test generation.

Constraints:
- The environment is Ubuntu and commands run from the repository root.
- Python {py_version} is requested.
- Use uv for virtual-environment creation and Python package installation when practical.
- Include creation and activation of .venv when needed.
- Install the repository itself plus runtime/test dependencies needed to import and exercise it.
- Include the requests package, which the GSO harness requires.
- Do not run installation commands and do not modify any files.
- Return only the structured install_commands result required by the response schema.
- Each array item must be one executable shell command, in execution order.
"""

    try:
        with tempfile.TemporaryDirectory(prefix="gso-codex-install-") as temp_dir:
            temp_path = Path(temp_dir)
            schema_path = temp_path / "schema.json"
            output_path = temp_path / "response.json"
            schema_path.write_text(
                json.dumps(CODEX_INSTALL_COMMAND_SCHEMA), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--color",
                    "never",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "--cd",
                    str(repo_path),
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=CODEX_INSTALL_COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()
                if len(detail) > 1000:
                    detail = detail[-1000:]
                logger.warning(
                    "Codex failed to infer install commands"
                    + (f": {detail}" if detail else "")
                )
                return None
            response = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        logger.warning(f"Codex failed to infer install commands: {error}")
        return None

    commands = response.get("install_commands") if isinstance(response, dict) else None
    if not isinstance(commands, list) or not commands:
        logger.warning("Codex returned no install commands")
        return None
    if any(not isinstance(command, str) or not command.strip() for command in commands):
        logger.warning("Codex returned invalid install commands")
        return None
    return [command.strip() for command in commands]


def update_yaml_install_commands(
    yaml_path: str | Path,
    install_commands: list[str],
) -> None:
    """Atomically replace ``install_commands`` in an experiment YAML file."""
    yaml_path = Path(yaml_path).expanduser().resolve()
    original = yaml_path.read_text(encoding="utf-8")
    config = yaml.safe_load(original)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment YAML must contain a mapping: {yaml_path}")

    rendered = yaml.dump(
        {"install_commands": list(install_commands)},
        Dumper=IndentDumper,
        sort_keys=False,
        allow_unicode=True,
    )
    lines = original.splitlines(keepends=True)
    key_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^install_commands\s*:", line)
        ),
        None,
    )
    if key_index is None:
        separator = "" if not original or original.endswith("\n") else "\n"
        updated = original + separator + rendered
    else:
        block_end = key_index + 1
        while block_end < len(lines):
            line = lines[block_end]
            if line.strip() and not line[0].isspace():
                break
            block_end += 1
        updated = "".join(lines[:key_index]) + rendered + "".join(lines[block_end:])

    file_mode = stat.S_IMODE(yaml_path.stat().st_mode)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=yaml_path.parent,
            prefix=f".{yaml_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(updated)
        os.chmod(temporary_path, file_mode)
        os.replace(temporary_path, yaml_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def log_generation_event(test_context: str, event: str) -> None:
    """Emit a structured lifecycle event that run_pipeline.py can relay."""
    print(f"{GENERATION_EVENT_PREFIX} {test_context} {event}", flush=True)


def create_generation_problem(
    repo: Repo, api: str, commits: list[PerformanceCommit], config: dict
) -> Problem:
    """Create a problem using configured install commands or model defaults."""
    return Problem.create_prob(repo, api, commits, config)


def _build_legacy_api_preflight_test(apis: list[str], repo_name: str) -> str:
    """Build the legacy import-only API smoke test.

    API analysis sometimes records a fully qualified name (``numpy.asarray``),
    but it can also record a class method without its module
    (``DataFrame.dropna``). The embedded resolver first tries qualified imports,
    then exported attributes from the installed distribution, and finally scans
    its sources before importing modules that define the recorded API.
    """
    apis_json = json.dumps(apis, ensure_ascii=False)
    repo_name_json = json.dumps(repo_name)
    prefix_json = json.dumps(API_PREFLIGHT_RESULT_PREFIX)
    return f"""import ast
import importlib
import importlib.metadata
import json
import timeit
from pathlib import Path

TARGET_APIS = {apis_json}
REPO_NAME = {repo_name_json}
RESULT_PREFIX = {prefix_json}
ALIASES = {{
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib.pyplot",
    "tf": "tensorflow",
    "torch": "torch",
}}


def _normalize_distribution_name(name):
    return name.lower().replace("-", "_").replace(".", "_")


def _getattr_chain(value, attributes):
    for attribute in attributes:
        value = getattr(value, attribute)
    return value


def _repository_roots():
    normalized_repo = _normalize_distribution_name(REPO_NAME)
    roots = {{normalized_repo}}
    try:
        distributions = importlib.metadata.packages_distributions()
    except Exception:
        distributions = {{}}
    for package, names in distributions.items():
        if any(_normalize_distribution_name(name) == normalized_repo for name in names):
            roots.add(package)

    repo_path = Path("/workspace") / REPO_NAME
    if repo_path.is_dir():
        for package_parent in (repo_path, repo_path / "src"):
            if not package_parent.is_dir():
                continue
            for child in package_parent.iterdir():
                if child.is_dir() and (child / "__init__.py").is_file():
                    roots.add(child.name)
    return sorted(roots)


def _module_defines_api(source_path, api_root):
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == api_root:
                return True
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == api_root for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == api_root:
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[-1]) == api_root for alias in node.names):
                return True
    return False


def _candidate_modules(root_module, api_root):
    candidates = []
    for raw_package_path in getattr(root_module, "__path__", []):
        package_path = Path(raw_package_path)
        for source_path in package_path.rglob("*.py"):
            if source_path.name == "__main__.py":
                continue
            if not _module_defines_api(source_path, api_root):
                continue
            relative = source_path.relative_to(package_path)
            suffix_parts = list(relative.with_suffix("").parts)
            if suffix_parts[-1] == "__init__":
                suffix_parts.pop()
            module_name = ".".join([root_module.__name__, *suffix_parts])
            candidates.append(module_name)
    return sorted(set(candidates), key=lambda name: (name.count("."), name))


def _resolve_api(api):
    parts = api.split(".")
    errors = []

    alias_module = ALIASES.get(parts[0])
    if alias_module is not None:
        try:
            return _getattr_chain(importlib.import_module(alias_module), parts[1:])
        except (Exception, SystemExit) as error:
            errors.append(f"{{alias_module}}: {{type(error).__name__}}: {{error}}")

    for split_index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            module = importlib.import_module(module_name)
            return _getattr_chain(module, parts[split_index:])
        except (Exception, SystemExit) as error:
            errors.append(f"{{module_name}}: {{type(error).__name__}}: {{error}}")

    roots = _repository_roots()
    for root_name in roots:
        try:
            root_module = importlib.import_module(root_name)
            return _getattr_chain(root_module, parts)
        except (Exception, SystemExit) as error:
            errors.append(f"{{root_name}}: {{type(error).__name__}}: {{error}}")
            try:
                root_module = importlib.import_module(root_name)
            except (Exception, SystemExit):
                continue

        for module_name in _candidate_modules(root_module, parts[0]):
            try:
                module = importlib.import_module(module_name)
                return _getattr_chain(module, parts)
            except (Exception, SystemExit):
                continue

    detail = "; ".join(errors[-8:])
    raise ImportError(
        f"could not resolve {{api!r}} from repository distribution roots {{roots}}"
        + (f"; recent errors: {{detail}}" if detail else "")
    )


def setup():
    return None


def experiment():
    results = {{}}
    for api in TARGET_APIS:
        try:
            value = _resolve_api(api)
            results[api] = {{
                "ok": True,
                "resolved": f"{{getattr(value, '__module__', '')}}."
                f"{{getattr(value, '__qualname__', getattr(value, '__name__', ''))}}",
            }}
        except (Exception, SystemExit) as error:
            message = f"{{type(error).__name__}}: {{error}}"
            results[api] = {{"ok": False, "error": message[-160:]}}
    print(RESULT_PREFIX + json.dumps(results, separators=(",", ":")), flush=True)
    failures = [api for api, result in results.items() if not result["ok"]]
    if failures:
        raise RuntimeError("API reference preflight failed: " + ", ".join(failures))
    return results


def store_result(result, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file)


def load_result(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def check_equivalence(reference, current):
    assert reference == current


def run_test(eqcheck=False, reference=False, prefix=""):
    setup()
    execution_time, result = timeit.timeit(experiment, number=1)
    result_path = f"{{prefix}}_api_preflight.json"
    if reference:
        store_result(result, result_path)
    elif eqcheck:
        check_equivalence(load_result(result_path), result)
    return execution_time
"""


def build_api_preflight_test(apis: list[str], repo_name: str) -> str:
    """Build a parent-revision API resolution and metadata probe."""
    return build_parent_api_preflight_test(
        apis,
        repo_name,
        API_PREFLIGHT_RESULT_PREFIX,
    )


def _parse_api_preflight_results(output: str | None) -> dict[str, dict] | None:
    """Extract the per-API report emitted by the parent-revision probe."""
    if not output:
        return None
    marker_index = output.rfind(API_PREFLIGHT_RESULT_PREFIX)
    if marker_index < 0:
        return None
    report = output[marker_index + len(API_PREFLIGHT_RESULT_PREFIX) :].splitlines()[0]
    try:
        parsed = json.loads(report)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _api_preflight_report(
    result: GeneratedTestExecutionResult,
) -> dict[str, dict] | None:
    """Load a complete probe report from its persisted Docker log."""
    if result.log_path:
        try:
            log_text = Path(result.log_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as error:
            logger.warning(
                "Failed to read API preflight log %s: %s", result.log_path, error
            )
        else:
            report = _parse_api_preflight_results(log_text)
            if report is not None:
                return report
    return _parse_api_preflight_results(result.error)


def _commit_date_sort_key(commit: PerformanceCommit) -> tuple[datetime, str]:
    """Return a deterministic oldest-first key, normalizing naive dates to UTC."""
    commit_date = commit.date
    if commit_date.tzinfo is None:
        commit_date = commit_date.replace(tzinfo=timezone.utc)
    return commit_date.astimezone(timezone.utc), commit.commit_hash


def _group_problems_by_commit(
    problems: list[Problem],
) -> list[tuple[PerformanceCommit, list[Problem]]]:
    groups: dict[str, tuple[PerformanceCommit, list[Problem]]] = {}
    for problem in problems:
        for commit in problem.commits:
            _, grouped_problems = groups.setdefault(commit.commit_hash, (commit, []))
            grouped_problems.append(problem)
    return sorted(groups.values(), key=lambda group: _commit_date_sort_key(group[0]))


def _api_preflight_scope(
    groups: list[tuple[PerformanceCommit, list[Problem]]],
) -> tuple[set[tuple[str, str]], set[str]]:
    pairs = {
        (problem.api, commit.commit_hash)
        for commit, grouped_problems in groups
        for problem in grouped_problems
    }
    return pairs, {api for api, _ in pairs}


def _print_api_preflight_plan(
    groups: list[tuple[PerformanceCommit, list[Problem]]],
) -> None:
    pairs, apis = _api_preflight_scope(groups)
    print(
        f"API preflight: testing {len(pairs)} API/commit pair(s) "
        f"({len(apis)} unique API(s)) across {len(groups)} commit(s)",
        flush=True,
    )


def _print_api_preflight_summary(
    groups: list[tuple[PerformanceCommit, list[Problem]]],
    accepted_pairs: set[tuple[str, str]],
) -> None:
    pairs, apis = _api_preflight_scope(groups)
    passed_apis = {api for api, _ in accepted_pairs}
    passed_commits = {commit_hash for _, commit_hash in accepted_pairs}
    print(
        f"API preflight complete: {len(accepted_pairs)}/{len(pairs)} "
        f"API/commit pair(s) passed; retained across "
        f"{len(passed_commits)}/{len(groups)} commit(s) and "
        f"{len(passed_apis)}/{len(apis)} unique API(s)",
        flush=True,
    )


def _run_api_preflight_for_commit(
    commit: PerformanceCommit,
    grouped_problems: list[Problem],
    config: GeneratedTestExecutionConfig,
    *,
    stage: str,
    task_index: int,
    task_count: int,
) -> tuple[set[str], list[dict], dict[str, dict]]:
    representative = grouped_problems[0]
    apis = [problem.api for problem in grouped_problems]
    task_prefix = f"API preflight task {task_index}/{task_count}"
    print(
        f"{task_prefix}: testing {len(apis)} API/commit pair(s) on "
        f"{commit.quick_hash()}...",
        flush=True,
    )
    test = build_api_preflight_test(apis, representative.repo.repo_name)
    try:
        result = evaluate_generated_test(representative, commit, test, config)
    except Exception as error:
        passed = False
        result_error = f"{type(error).__name__}: {error}"
        report = None
    else:
        passed = result.passed
        result_error = result.error
        report = _api_preflight_report(result)

    if passed and config.prepared_commit_hash is not None:
        missing_reports = [
            api
            for api in apis
            if report is None
            or not isinstance(report.get(api), dict)
            or report[api].get("ok") is not True
        ]
        if missing_reports:
            passed = False
            result_error = (
                "Parent API context probe produced no successful metadata for: "
                + ", ".join(missing_reports)
            )

    passed_apis: set[str] = set()
    api_contexts: dict[str, dict] = {}
    diagnostics = []
    if report is not None:
        for api in apis:
            api_report = report.get(api)
            if isinstance(api_report, dict) and api_report.get("ok") is True:
                api_contexts[api] = {
                    key: value for key, value in api_report.items() if key != "ok"
                }

    if passed:
        passed_apis.update(apis)
    else:
        problems_by_api = {problem.api: problem for problem in grouped_problems}
        for api in apis:
            api_report = report.get(api) if report is not None else None
            if isinstance(api_report, dict) and api_report.get("ok") is True:
                passed_apis.add(api)
                continue
            error = (
                (api_report.get("error") or result_error)
                if isinstance(api_report, dict)
                else result_error
            )
            error = error or "API preflight failed without diagnostics"
            diagnostics.append(
                {
                    "pid": problems_by_api[api].pid,
                    "api": api,
                    "commit_hash": commit.commit_hash,
                    "stage": stage,
                    "error": error,
                }
            )
            logger.error(
                "Skipping API/commit %s/%s after preflight failure: %s",
                api,
                commit.quick_hash(),
                error,
            )
    print(
        f"{task_prefix}: {len(passed_apis)}/{len(apis)} API/commit pair(s) "
        f"passed on {commit.quick_hash()}",
        flush=True,
    )
    return passed_apis, diagnostics, api_contexts


def _filter_problem_commit_pairs(
    problems: list[Problem], accepted_pairs: set[tuple[str, str]]
) -> list[Problem]:
    for problem in problems:
        problem.commits = [
            commit
            for commit in problem.commits
            if (problem.api, commit.commit_hash) in accepted_pairs
        ]
    return [problem for problem in problems if problem.commits]


def preflight_generation_problems(
    problems: list[Problem],
    config: GeneratedTestExecutionConfig | None,
) -> tuple[list[Problem], list[dict]]:
    """Preflight commit groups and filter only failing API/commit pairs."""
    if config is None:
        return problems, []

    groups = _group_problems_by_commit(problems)
    _print_api_preflight_plan(groups)
    accepted_pairs: set[tuple[str, str]] = set()
    diagnostics: list[dict] = []
    for task_index, (commit, grouped_problems) in enumerate(groups, start=1):
        passed_apis, group_diagnostics, _ = _run_api_preflight_for_commit(
            commit,
            grouped_problems,
            config,
            stage="install_api_preflight",
            task_index=task_index,
            task_count=len(groups),
        )
        accepted_pairs.update((api, commit.commit_hash) for api in passed_apis)
        diagnostics.extend(group_diagnostics)
    _print_api_preflight_summary(groups, accepted_pairs)
    return _filter_problem_commit_pairs(problems, accepted_pairs), diagnostics


def prepare_and_preflight_generation_problems(
    problems: list[Problem],
    config: GeneratedTestExecutionConfig,
    *,
    num_workers: int,
    session_id: str,
) -> tuple[
    list[Problem],
    list[dict],
    dict[str, GeneratedTestExecutionConfig],
]:
    """Prepare/install and API-preflight unique commits concurrently."""
    groups = _group_problems_by_commit(problems)
    _print_api_preflight_plan(groups)
    accepted_pairs: set[tuple[str, str]] = set()
    diagnostics: list[dict] = []
    execution_configs: dict[str, GeneratedTestExecutionConfig] = {}

    def prepare_group(
        task_index: int,
        commit: PerformanceCommit,
        grouped_problems: list[Problem],
    ) -> tuple[
        PerformanceCommit,
        set[str],
        list[dict],
        GeneratedTestExecutionConfig | None,
    ]:
        representative = grouped_problems[0]
        apis = [problem.api for problem in grouped_problems]
        task_prefix = f"API preflight task {task_index}/{len(groups)}"
        print(
            f"{task_prefix}: preparing {len(apis)} API/commit pair(s) on "
            f"{commit.quick_hash()}...",
            flush=True,
        )
        try:
            prepared_config = prepare_commit_test_execution(
                representative,
                commit,
                config,
                session_id=session_id,
            )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            group_diagnostics = [
                {
                    "pid": problem.pid,
                    "api": problem.api,
                    "commit_hash": commit.commit_hash,
                    "stage": "commit_environment_prepare",
                    "error": detail,
                }
                for problem in grouped_problems
            ]
            logger.error(
                "Skipping commit %s after environment preparation failure: %s",
                commit.quick_hash(),
                detail,
            )
            print(
                f"{task_prefix}: 0/{len(apis)} API/commit pair(s) passed on "
                f"{commit.quick_hash()} (environment preparation failed)",
                flush=True,
            )
            return commit, set(), group_diagnostics, None

        _register_prepared_execution(prepared_config)
        passed_apis, group_diagnostics, api_contexts = _run_api_preflight_for_commit(
            commit,
            grouped_problems,
            prepared_config,
            stage="api_preflight",
            task_index=task_index,
            task_count=len(groups),
        )
        prepared_config = replace(prepared_config, api_contexts=api_contexts)
        if not passed_apis:
            _cleanup_prepared_execution(prepared_config)
            prepared_config = None
        return commit, passed_apis, group_diagnostics, prepared_config

    workers = min(max(1, num_workers), len(groups), MAX_COMMIT_PREPARATION_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(prepare_group, task_index, commit, grouped_problems)
            for task_index, (commit, grouped_problems) in enumerate(groups, start=1)
        ]
        for future in as_completed(futures):
            commit, passed_apis, group_diagnostics, prepared_config = future.result()
            diagnostics.extend(group_diagnostics)
            accepted_pairs.update((api, commit.commit_hash) for api in passed_apis)
            if prepared_config is not None:
                execution_configs[commit.commit_hash] = prepared_config

    _print_api_preflight_summary(groups, accepted_pairs)
    filtered = _filter_problem_commit_pairs(problems, accepted_pairs)
    return filtered, diagnostics, execution_configs


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
    scenarios: list[str] = field(default_factory=list)
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


def add_scenario_comment(test: str, scenario: str) -> str:
    """Persist a short scenario alongside its generated test."""
    normalized = " ".join(scenario.split())
    return f"# GSO generated scenario: {normalized}\n\n" + test.lstrip()


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


@dataclass
class BatchTestSlotState:
    """Latest generated candidate and validation state for one stable slot."""

    slot: int
    attempts: int = 0
    status: str = "pending"
    scenario: str | None = None
    code: str | None = None
    generated_test: str | None = None
    error: str | None = None
    execution_log: str | None = None


def _batch_generation_state(states: dict[int, BatchTestSlotState]) -> str:
    """Return all successful and failed latest tests for a repair request."""
    successful_tests = []
    failed_tests = []
    for slot in sorted(states):
        state = states[slot]
        item = {
            "slot": slot,
            "scenario": state.scenario,
            "code": state.code,
        }
        if state.status == "accepted":
            successful_tests.append(item)
        else:
            failed_tests.append(
                {
                    **item,
                    "attempt": state.attempts,
                    "failure": {
                        "summary": state.error or "No valid test was returned",
                        "execution_log": state.execution_log,
                    },
                }
            )
    return json.dumps(
        {
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
        },
        indent=2,
        ensure_ascii=False,
    )


def _record_batch_attempt(
    result: CommitGenerationResult,
    state: BatchTestSlotState,
    *,
    round_number: int,
    error: str | None,
) -> None:
    result.test_attempts.append(
        {
            "scenario_index": state.slot,
            "slot": state.slot,
            "retry_number": round_number,
            "attempt": state.attempts,
            "scenario": state.scenario,
            "code": state.code,
            "raw_output": None,
            "error": error,
            "execution_log": state.execution_log,
        }
    )


def generate_commit_tests(task: CommitGenerationTask) -> CommitGenerationResult:
    """Generate tests in batches and repair only failed stable slots."""
    generation_started = time.monotonic()
    llm_seconds = 0.0
    validation_seconds = 0.0
    result = CommitGenerationResult(
        problem_index=task.problem_index,
        commit_index=task.commit_index,
        pid=task.problem.pid,
        commit_hash=task.commit.commit_hash,
    )
    states = {
        slot: BatchTestSlotState(slot=slot)
        for slot in range(1, task.args.n + 1)
    }

    try:
        prepare_args: tuple = (task.repo, task.problem, task.commit, False)
        if task.execution_config is not None and task.execution_config.api_contexts:
            api_context = task.execution_config.api_contexts.get(task.problem.api)
            if api_context is not None:
                prepare_args += (api_context,)
        commit_tests = prepare_mp_helper(prepare_args)

        session_context = (
            open_generated_test_validation_session(
                task.problem,
                task.commit,
                task.execution_config,
            )
            if task.execution_config is not None
            else nullcontext(None)
        )
        with session_context as validation_session:
            for round_number in range(SEMANTIC_RETRY_COUNT + 1):
                pending_slots = [
                    slot
                    for slot, state in states.items()
                    if state.status == "pending"
                    and state.attempts < SEMANTIC_RETRY_COUNT + 1
                ]
                if not pending_slots:
                    break

                if round_number == 0:
                    task_message = BATCH_SCENARIO_TEST_MSG.format(
                        api=task.problem.api,
                        repo_name=task.repo.repo_name,
                        test_count=len(pending_slots),
                        slot_list=pending_slots,
                    )
                else:
                    task_message = REPAIR_BATCH_TEST_MSG.format(
                        api=task.problem.api,
                        repo_name=task.repo.repo_name,
                        generation_state=_batch_generation_state(states),
                        first_slot=pending_slots[0],
                        failed_count=len(pending_slots),
                        failed_slots=pending_slots,
                    )
                if task.repo.repo_instr:
                    task_message += (
                        "\n\nRepo-specific Instructions:\n"
                        f"{task.repo.repo_instr}\n"
                    )

                messages = _prepare_generation_messages(
                    commit_tests.chat_messages, task_message
                )
                payload = prepare_model_payload(task.args.model_name, messages)
                request_args = (
                    task.args
                    if round_number == 0
                    else task.args.model_copy(update={"use_cache": False})
                )
                event_context = (
                    f"target {task.target_index + 1}/{task.target_count} | "
                    f"{task.problem.pid}/{task.commit.quick_hash()} | "
                    f"batch {round_number + 1}/{SEMANTIC_RETRY_COUNT + 1} |"
                )
                for slot in pending_slots:
                    states[slot].attempts += 1

                raw_output = None
                try:
                    action = (
                        f"requesting {len(pending_slots)} tests"
                        if round_number == 0
                        else f"requesting repairs for slots {pending_slots}"
                    )
                    log_generation_event(event_context, action)
                    completion_started = time.monotonic()
                    try:
                        raw_output = get_llm_completion(
                            request_args,
                            payload,
                            stream=request_args.stream,
                        )
                    finally:
                        llm_seconds += time.monotonic() - completion_started
                    result.test_outputs.append(raw_output)
                    parsed_batch = parse_generated_test_batch(
                        raw_output,
                        expected_slots=set(pending_slots),
                        context=(
                            f"{task.problem.pid}/{task.commit.quick_hash()} "
                            f"generation batch {round_number + 1}"
                        ),
                    )
                except Exception as error:
                    detail = (
                        f"LLM batch generation failed: {type(error).__name__}: {error}"
                    )
                    for slot in pending_slots:
                        state = states[slot]
                        state.error = detail
                        state.execution_log = None
                        _record_batch_attempt(
                            result,
                            state,
                            round_number=round_number,
                            error=detail,
                        )
                    parsed_batch = None

                runnable: list[BatchTestSlotState] = []
                if parsed_batch is not None:
                    accepted_scenarios = {
                        " ".join(state.scenario.split()).casefold()
                        for state in states.values()
                        if state.status == "accepted" and state.scenario
                    }
                    accepted_codes = {
                        state.code.strip()
                        for state in states.values()
                        if state.status == "accepted" and state.code
                    }
                    for slot in pending_slots:
                        state = states[slot]
                        parse_error = parsed_batch.errors.get(slot)
                        if parse_error is not None:
                            state.error = parse_error
                            state.execution_log = None
                            _record_batch_attempt(
                                result,
                                state,
                                round_number=round_number,
                                error=parse_error,
                            )
                            continue

                        item = parsed_batch.items[slot]
                        state.scenario = item.scenario
                        state.code = item.code
                        state.execution_log = None
                        normalized_scenario = " ".join(
                            item.scenario.split()
                        ).casefold()
                        normalized_code = item.code.strip()
                        if normalized_scenario in accepted_scenarios:
                            validation_error = (
                                f"slot {slot}: scenario duplicates an accepted test"
                            )
                        elif normalized_code in accepted_codes:
                            validation_error = (
                                f"slot {slot}: code duplicates an accepted test"
                            )
                        else:
                            try:
                                validate_generated_test(
                                    item.code,
                                    context=f"slot {slot} generated test",
                                )
                            except GeneratedTestError as error:
                                validation_error = str(error)
                            else:
                                validation_error = None

                        if validation_error is not None:
                            state.error = validation_error
                            _record_batch_attempt(
                                result,
                                state,
                                round_number=round_number,
                                error=validation_error,
                            )
                            continue
                        state.generated_test = add_scenario_comment(
                            item.code, item.scenario
                        )
                        accepted_scenarios.add(normalized_scenario)
                        accepted_codes.add(normalized_code)
                        runnable.append(state)

                for state in runnable:
                    slot_context = (
                        f"target {task.target_index + 1}/{task.target_count} | "
                        f"{task.problem.pid}/{task.commit.quick_hash()} | "
                        f"test {state.slot}/{task.args.n} |"
                    )
                    execution_error = None
                    if validation_session is not None:
                        log_generation_event(
                            slot_context,
                            f"starting test (attempt {state.attempts}/"
                            f"{SEMANTIC_RETRY_COUNT + 1})",
                        )
                        validation_started = time.monotonic()
                        try:
                            execution_result = validation_session.validate(
                                state.generated_test,
                                slot=state.slot,
                                attempt=state.attempts,
                            )
                        except Exception as error:
                            execution_error = (
                                "Automated execution raised "
                                f"{type(error).__name__}: {error}"
                            )
                        else:
                            state.execution_log = execution_result.log_path
                            if not execution_result.passed:
                                execution_error = execution_result.error or (
                                    "Generated test execution failed without diagnostics"
                                )
                        finally:
                            validation_seconds += time.monotonic() - validation_started

                    if execution_error is not None:
                        state.error = execution_error
                        log_generation_event(
                            slot_context,
                            "test failed"
                            + (
                                f" (log: {state.execution_log})"
                                if state.execution_log
                                else ""
                            ),
                        )
                        _record_batch_attempt(
                            result,
                            state,
                            round_number=round_number,
                            error=execution_error,
                        )
                        continue

                    state.status = "accepted"
                    state.error = None
                    log_generation_event(slot_context, "test passed")
                    _record_batch_attempt(
                        result,
                        state,
                        round_number=round_number,
                        error=None,
                    )

                for slot in pending_slots:
                    state = states[slot]
                    if (
                        state.status != "accepted"
                        and state.attempts >= SEMANTIC_RETRY_COUNT + 1
                    ):
                        state.status = "exhausted"

                accepted_count = sum(
                    state.status == "accepted" for state in states.values()
                )
                exhausted_count = sum(
                    state.status == "exhausted" for state in states.values()
                )
                elapsed_seconds = time.monotonic() - generation_started
                print(
                    f"{GENERATION_PROGRESS_PREFIX} "
                    f"target {task.target_index + 1}/{task.target_count} | "
                    f"{task.problem.pid}/{task.commit.quick_hash()} | "
                    f"batch {round_number + 1}/{SEMANTIC_RETRY_COUNT + 1} | "
                    f"requested={len(pending_slots)} | accepted={accepted_count} | "
                    f"exhausted={exhausted_count} | "
                    f"elapsed={elapsed_seconds:.1f}s | llm={llm_seconds:.1f}s | "
                    f"validation={validation_seconds:.1f}s",
                    flush=True,
                )

        accepted_states = [
            states[slot]
            for slot in sorted(states)
            if states[slot].status == "accepted"
        ]
        result.scenarios = [state.scenario for state in accepted_states]
        result.failed_test_slots = [
            {
                "scenario_index": state.slot,
                "slot": state.slot,
                "attempts": state.attempts,
                "scenario": state.scenario,
                "error": state.error,
                "execution_log": state.execution_log,
            }
            for state in states.values()
            if state.status != "accepted"
        ]
        if not accepted_states:
            raise GeneratedTestError(
                f"{task.problem.pid}/{task.commit.quick_hash()}: no valid tests were "
                f"generated after attempting {task.args.n} test slot(s)"
            )
        commit_tests.add_samples(
            [state.generated_test for state in accepted_states]
        )
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
        self.exp_dir = EXPS_DIR / self.exp_id

        # set repo-specific instructions
        if "repo_instr" in self.config:
            self.repo.repo_instr = self.config["repo_instr"]

        self.ensure_install_commands(args)
        self.candidates = self.get_commit_map(self.repo)

    def ensure_install_commands(self, args) -> None:
        """Infer and persist repository-specific commands when YAML leaves them empty."""
        if self.config.get("install_commands"):
            return

        if args.docker_repo_path is not None:
            repo_path = Path(args.docker_repo_path).expanduser()
        else:
            repo_path = Path(self.repo.local_repo_path)

        commands = request_install_commands_from_codex(
            repo_path,
            str(self.config.get("py_version", "3.9")),
        )
        if commands is None:
            logger.warning(
                "Using Problem default install commands because Codex inference failed"
            )
            return

        source_path = Path(args.yaml_path).expanduser().resolve()
        copied_path = (self.exp_dir / f"{self.exp_id}.yaml").resolve()
        yaml_paths = [source_path]
        if copied_path != source_path:
            yaml_paths.append(copied_path)
        for yaml_path in yaml_paths:
            update_yaml_install_commands(yaml_path, commands)
        self.config["install_commands"] = list(commands)
        print(
            f"Codex inferred {len(commands)} install command(s) from {repo_path}; "
            f"updated {source_path}",
            flush=True,
        )

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
                docker_cache_dir=args.docker_cache_dir,
                rebuild_docker_image=args.rebuild_docker_image,
                keep_containers=args.keep_containers,
                keep_workspaces=args.keep_workspaces,
            ),
        )
        generation_session_id = uuid.uuid4().hex[:12]
        if execution_config is None:
            # Test-only/no-execution mode: retain the previous behavior without
            # creating prepared Docker environments.
            problems, preflight_diagnostics = preflight_generation_problems(
                problems, None
            )
            commit_execution_configs = {
                commit.commit_hash: None
                for problem in problems
                for commit in problem.commits
            }
        else:
            problems, preflight_diagnostics, commit_execution_configs = (
                prepare_and_preflight_generation_problems(
                    problems,
                    execution_config,
                    num_workers=args.multiprocess,
                    session_id=generation_session_id,
                )
            )
        if not problems:
            retry_path = self.save_retry_diagnostics(preflight_diagnostics, args)
            detail = (
                f" Diagnostics were saved to {retry_path}."
                if retry_path is not None
                else ""
            )
            print(
                "No APIs passed install/import preflight; nothing to generate."
                + detail,
                flush=True,
            )
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
        # Submit generation workers oldest-first as well. With multiple workers,
        # commits can overlap, but the queue still favors nearby earlier revisions.
        commit_tasks.sort(key=lambda task: _commit_date_sort_key(task.commit))
        commit_tasks = [
            replace(
                task,
                target_index=target_index,
                target_count=len(commit_tasks),
                execution_config=commit_execution_configs[task.commit.commit_hash],
            )
            for target_index, task in enumerate(commit_tasks)
        ]

        commit_workers = min(args.multiprocess, len(commit_tasks))
        generation_commit_count = len(
            {task.commit.commit_hash for task in commit_tasks}
        )
        print(
            f"Generating {len(problems)} problems for {len(commit_tasks)} "
            f"API/commit task(s) across {generation_commit_count} commit(s) "
            f"(tests_per_commit={args.n}, tests_per_initial_request={args.n}, "
            f"commit_workers={commit_workers}, max_tokens={args.max_tokens}, "
            f"stream={args.stream})"
        )
        generation_outputs = generate_commits_as_completed(commit_tasks, commit_workers)

        generation_errors = []
        failure_diagnostics = list(preflight_diagnostics)
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

        for prepared_config in commit_execution_configs.values():
            if prepared_config is not None:
                _cleanup_prepared_execution(prepared_config)

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
