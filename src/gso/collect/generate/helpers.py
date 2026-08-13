import ast
import json
import os
import re
from collections.abc import Sequence

import tiktoken
from ghapi.core import GhApi
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gso.data import Repo

tokenizer = tiktoken.encoding_for_model("gpt-4")


def count_tokens(context: str):
    return len(tokenizer.encode(context, disallowed_special=()))


def get_github_convo(repo: Repo, pr_num: str, max_count=5) -> str:
    """Get the conversation for a pull request.

    Goal: capture any information that might be related to testing the PR's performance.

    Note: this is not PR reviews, but regular comments on the PR.
    PR reviews usually don't have interesting testing information and also
    contain older code edits that are not relevant to the state after PR merge.
    """

    def format_comments(comments, max_count=5, min_lines=2):
        """Formats the comments for a pull request."""
        formatted_comments = []
        for comment in comments:
            if comment.user.type != "User":
                continue

            if len(comment.body.split("\n")) < min_lines:
                continue

            body = re.sub("![.*](.*)", "", comment.body)
            body = re.sub("<img.*>", "", body).strip()
            formatted_comment = f"{comment.user.login}: {body}"
            formatted_comment = strip_empty_lines(formatted_comment)
            formatted_comments.append(f"\n{formatted_comment.strip()}")

        return "".join(formatted_comments[:max_count])

    repo_owner = repo.repo_owner
    repo_name = repo.repo_name

    # Use ghapi to get PR discussion messages. Explicitly disable authentication
    # when no token is configured to avoid one warning per parallel task.
    github_token = os.getenv("GHAPI_TOKEN") or os.getenv("GITHUB_TOKEN")
    api = GhApi(token=github_token, authenticate=bool(github_token))
    try:
        pr = api.pulls.get(repo_owner, repo_name, int(pr_num))
        comments = api.issues.list_comments(repo_owner, repo_name, int(pr_num))
        comments_str = format_comments(comments)
    except Exception as e:
        return ""

    resp = ""
    if pr and pr.body and pr.body != "":
        resp += f"Description: {pr.body.strip()}"
    if comments_str != "":
        resp += f"\n\nComments:\n{comments_str.strip()}"

    return resp


def strip_empty_lines(text: str):
    return "\n".join([line for line in text.splitlines() if line.strip()])


REQUIRED_TEST_FUNCTIONS = frozenset(
    {
        "setup",
        "experiment",
        "store_result",
        "load_result",
        "check_equivalence",
        "run_test",
    }
)
_CODE_FENCE_RE = re.compile(
    r"```[ \t]*(?P<language>[^\r\n`]*)\r?\n(?P<code>.*?)```", re.DOTALL
)


class GeneratedTestError(ValueError):
    """Raised when an LLM completion is not a usable performance test."""


class GeneratedScenarioError(GeneratedTestError):
    """Raised when an LLM completion is not a usable scenario plan."""


class TestScenario(BaseModel):
    """Structured plan used to generate one performance test."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1)
    workload: str = Field(min_length=1)
    input_characteristics: str = Field(min_length=1)
    api_usage: str = Field(min_length=1)
    optimization_focus: str = Field(min_length=1)
    equivalence_strategy: str = Field(min_length=1)
    distinguishing_factor: str = Field(min_length=1)


def _format_validation_errors(title: str, errors: list[str]) -> str:
    displayed = errors[:20]
    details = "\n".join(f"- {error}" for error in displayed)
    if len(errors) > len(displayed):
        details += f"\n- ... and {len(errors) - len(displayed)} more error(s)"
    return f"{title}:\n{details}"


def extract_json_block(output: str, *, context: str = "scenario response") -> str:
    """Extract a JSON object from a raw or fenced model response."""
    if not isinstance(output, str) or not output.strip():
        raise GeneratedScenarioError(f"{context}: model returned an empty response")

    fence_count = output.count("```")
    if fence_count % 2:
        raise GeneratedScenarioError(
            f"{context}: response contains an unclosed code fence; it was likely "
            "truncated"
        )

    matches = list(_CODE_FENCE_RE.finditer(output))
    if matches:
        json_blocks = []
        for match in matches:
            language = match.group("language").strip().lower()
            if not language or language == "json":
                json_blocks.append(match.group("code").strip())
        if not json_blocks:
            raise GeneratedScenarioError(
                f"{context}: response contains code fences, but no JSON block"
            )
        content = max(json_blocks, key=len)
    elif fence_count:
        raise GeneratedScenarioError(
            f"{context}: response contains malformed code fences"
        )
    else:
        content = output.strip()

    if not content:
        raise GeneratedScenarioError(f"{context}: extracted JSON is empty")
    return content


def get_generated_scenarios(
    output: str,
    *,
    expected_count: int,
    context: str = "scenario response",
) -> list[TestScenario]:
    """Parse and validate the scenarios returned in one model choice."""
    content = extract_json_block(output, context=context)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GeneratedScenarioError(
            f"{context}: invalid JSON at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    if not isinstance(parsed, dict):
        raise GeneratedScenarioError(
            f"{context}: expected a JSON object with a 'scenarios' array"
        )
    raw_scenarios = parsed.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise GeneratedScenarioError(
            f"{context}: expected 'scenarios' to be a JSON array"
        )
    if len(raw_scenarios) != expected_count:
        raise GeneratedScenarioError(
            f"{context}: expected {expected_count} scenario(s), "
            f"got {len(raw_scenarios)}"
        )

    scenarios = []
    for index, raw_scenario in enumerate(raw_scenarios):
        try:
            scenarios.append(TestScenario.model_validate(raw_scenario))
        except ValidationError as exc:
            raise GeneratedScenarioError(
                f"{context}, scenario {index + 1}: invalid scenario structure: {exc}"
            ) from exc
    return scenarios


def extract_codeblock(output: str, *, context: str = "model response") -> str:
    """Extract Python from a completion without silently accepting truncation."""
    if not isinstance(output, str) or not output.strip():
        raise GeneratedTestError(f"{context}: model returned an empty response")

    fence_count = output.count("```")
    if fence_count % 2:
        raise GeneratedTestError(
            f"{context}: response contains an unclosed code fence; it was likely "
            "truncated (increase llm.max_tokens)"
        )

    matches = list(_CODE_FENCE_RE.finditer(output))
    if matches:
        python_blocks = []
        for match in matches:
            language = match.group("language").strip().lower()
            if not language or language in {"py", "python", "python3"}:
                python_blocks.append(match.group("code").strip())
        if not python_blocks:
            raise GeneratedTestError(
                f"{context}: response contains code fences, but no Python code block"
            )
        code = max(python_blocks, key=len)
    elif fence_count:
        raise GeneratedTestError(f"{context}: response contains malformed code fences")
    else:
        # Some OpenAI-compatible models ignore the requested Markdown wrapper.
        # Accept raw output only if the structural validation below succeeds.
        code = output.strip()

    if not code:
        raise GeneratedTestError(f"{context}: extracted Python code is empty")
    return code


def validate_generated_test(code: str, *, context: str = "generated test") -> None:
    """Validate the contract required by TIMEIT_TEMPLATE and TEST_HARNESS."""
    if not isinstance(code, str) or not code.strip():
        raise GeneratedTestError(f"{context}: generated Python code is empty")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        raise GeneratedTestError(
            f"{context}: invalid Python syntax at {location}: {exc.msg}"
        ) from exc

    issues = []
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    missing_functions = sorted(REQUIRED_TEST_FUNCTIONS - functions.keys())
    if missing_functions:
        issues.append("missing required function(s): " + ", ".join(missing_functions))

    imports_timeit = any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "timeit" and alias.asname in {None, "timeit"}
            for alias in node.names
        )
        for node in tree.body
    )
    if not imports_timeit:
        issues.append(
            "missing top-level 'import timeit' required by the appended timeit template"
        )

    if "run_test" in functions:
        run_test = functions["run_test"]
        positional_args = [*run_test.args.posonlyargs, *run_test.args.args]
        expected_args = ["eqcheck", "reference", "prefix"]
        actual_args = [arg.arg for arg in positional_args[: len(expected_args)]]
        if actual_args != expected_args:
            issues.append(
                "run_test must start with arguments "
                f"{tuple(expected_args)}, got {tuple(actual_args)}"
            )

    if issues:
        raise GeneratedTestError(f"{context}: " + "; ".join(issues))


def get_generated_test(output: str, *, context: str = "model response") -> str:
    """Extract and validate one generated performance test."""
    code = extract_codeblock(output, context=context)
    validate_generated_test(code, context=context)
    return code


def get_generated_tests(outputs: Sequence[Sequence[str]]) -> list[list[str]]:
    """Extract and validate every generated completion before mutating problems."""
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise GeneratedTestError("LLM output must be a sequence of completion groups")

    results: list[list[str]] = []
    errors: list[str] = []
    for group_index, output in enumerate(outputs):
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            errors.append(f"completion group {group_index + 1}: expected a sequence")
            results.append([])
            continue
        if not output:
            errors.append(f"completion group {group_index + 1}: no samples returned")
            results.append([])
            continue

        code_blocks = []
        for sample_index, sample in enumerate(output):
            context = f"completion group {group_index + 1}, sample {sample_index + 1}"
            try:
                code_blocks.append(get_generated_test(sample, context=context))
            except GeneratedTestError as exc:
                errors.append(str(exc))
        results.append(code_blocks)

    if errors:
        raise GeneratedTestError(
            _format_validation_errors(
                f"Rejected {len(errors)} invalid generated test completion(s)", errors
            )
        )
    return results


def validate_generated_test_counts(
    results: Sequence[Sequence[str]],
    *,
    expected_groups: int,
    expected_samples: int,
) -> None:
    errors = []
    if len(results) != expected_groups:
        errors.append(
            f"expected {expected_groups} completion group(s), got {len(results)}"
        )
    for index, samples in enumerate(results):
        if len(samples) != expected_samples:
            errors.append(
                f"completion group {index + 1}: expected {expected_samples} sample(s), "
                f"got {len(samples)}"
            )
    if errors:
        raise GeneratedTestError(
            _format_validation_errors("Unexpected LLM completion count", errors)
        )


def validate_problem_test_samples(problems: Sequence) -> None:
    """Fail before execution when a problems file contains malformed tests."""
    errors = []
    for problem in problems:
        if not problem.tests:
            errors.append(f"{problem.pid}: no commit tests were generated")
            continue
        for commit_tests in problem.tests:
            if not commit_tests.samples:
                errors.append(
                    f"{problem.pid}/{commit_tests.quick_hash}: no test samples were generated"
                )
                continue
            for sample_index, sample in enumerate(commit_tests.samples):
                context = (
                    f"{problem.pid}/{commit_tests.quick_hash}/test_{sample_index}.py"
                )
                try:
                    validate_generated_test(sample, context=context)
                except GeneratedTestError as exc:
                    errors.append(str(exc))
    if errors:
        raise GeneratedTestError(
            _format_validation_errors(
                "Invalid generated tests; regenerate the problems file before execution",
                errors,
            )
        )


def get_latest_good_tests(prob, commit_hash) -> list[str]:
    latest_run_key, latest_results = list(prob.results.items())[-1]
    good_test_ids = [r["test_id"] for r in latest_results if r["commit"] == commit_hash]
    good_tests = [prob.get_test(commit_hash, tid) for tid in good_test_ids]
    return good_tests
