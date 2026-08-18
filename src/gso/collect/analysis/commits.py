from __future__ import annotations

import os
import re
import json
import argparse
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from gso.data import PerformanceCommit, PerfAnalysis
from gso.collect.analysis.parser import CommitParser
from gso.collect.analysis.prompt import *
from gso.collect.analysis.utils import *
from gso.constants import *
from gso.utils.io import *
from gso.utils.llm import (
    configure_openai_compatible_llm,
    get_streaming_llm_completions,
)

if TYPE_CHECKING:
    from gso.collect.analysis.retriever import Retriever

GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")
MAX_COMMIT_TOKENS = 20000
MAX_OAI_TOKENS = 90000
THRESHOLD = 200
SKIP_API_ANALYSIS = False  # Set to True to skip API analysis
DEFAULT_ANALYSIS_MODEL = "o3-mini"
DEFAULT_LLM_MULTIPROCESS = 60
DEFAULT_LLM_CACHE_SETTINGS = {
    "commit_filter": True,
    "affected_files": True,
    "api_identification": True,
}


class PerfCommitAnalyzer:
    model_name = DEFAULT_ANALYSIS_MODEL
    llm_multiprocess = DEFAULT_LLM_MULTIPROCESS
    llm_max_tokens: int | None = None
    llm_openai_timeout: int | None = None
    llm_cache_settings = DEFAULT_LLM_CACHE_SETTINGS.copy()

    @classmethod
    def configure_llm(cls, config: dict) -> None:
        """Configure an OpenAI-compatible endpoint and per-stage caching."""
        llm_config = config.get("llm", {})
        if llm_config is None:
            llm_config = {}
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

        configured = configure_openai_compatible_llm(
            config,
            default_model=DEFAULT_ANALYSIS_MODEL,
            default_multiprocess=DEFAULT_LLM_MULTIPROCESS,
            model_env="GSO_ANALYSIS_MODEL",
            purpose="analysis",
        )
        cls.model_name = configured.model_name
        cls.llm_multiprocess = configured.multiprocess
        cls.llm_max_tokens = configured.max_tokens
        cls.llm_openai_timeout = configured.openai_timeout
        cls.llm_cache_settings = {
            stage: cache_config.get(stage, default)
            for stage, default in DEFAULT_LLM_CACHE_SETTINGS.items()
        }

    @classmethod
    def build_llm_args(cls, *, cache_stage: str, default_max_tokens: int):
        """Build analysis request arguments with experiment-wide LLM overrides."""
        from r2e.llms.llm_args import LLMArgs

        kwargs = {
            "model_name": cls.model_name,
            "cache_batch_size": 100,
            "multiprocess": cls.llm_multiprocess,
            "use_cache": cls.llm_cache_settings[cache_stage],
            "max_tokens": (
                cls.llm_max_tokens
                if cls.llm_max_tokens is not None
                else default_max_tokens
            ),
        }
        if cls.llm_openai_timeout is not None:
            kwargs["openai_timeout"] = cls.llm_openai_timeout
        return LLMArgs(**kwargs)

    @staticmethod
    def parse_diff_for_stats(
        commit: PerformanceCommit, repo_path: Path
    ) -> dict[str, int]:
        parser = CommitParser()
        diff = parser.parse_commit(
            commit.old_commit_hash,
            commit.commit_hash,
            commit.diff_text,
            commit.message,
            commit.date,
            repo_path,
        )

        stats = {
            "num_test_files": diff.num_test_files,
            "num_non_test_files": diff.num_non_test_files,
            "only_test_files": diff.num_files == diff.num_test_files,
            "only_non_test_files": diff.num_files == diff.num_non_test_files,
            "num_files": diff.num_files,
            "num_hunks": diff.num_hunks,
            "num_edited_lines": diff.num_edited_lines,
            "num_non_test_edited_lines": diff.num_non_test_edited_lines,
            "commit_year": diff.commit_date.year,
        }

        return stats

    @staticmethod
    def process_commit(
        commit_hash: str, repo_path: Path, max_year: int | None
    ) -> PerformanceCommit:
        # commit subject
        subject = run_git_command(
            ["git", "show", "--no-patch", "--format=%s", commit_hash], cwd=repo_path
        )

        # commit message
        message = run_git_command(
            ["git", "show", "--no-patch", "--format=%B", commit_hash], cwd=repo_path
        )

        # commit date
        date_str = run_git_command(
            ["git", "show", "-s", "--format=%cd", commit_hash], cwd=repo_path
        )
        date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y %z")

        if max_year and date.year <= max_year:
            return None

        # changed files
        files_changed = run_git_command(
            ["git", "show", "--name-only", "--format=", commit_hash], cwd=repo_path
        ).split("\n")

        # commit diff
        old_commit_hash = f"{commit_hash}^"

        try:
            diff_text = run_git_command(
                ["git", "diff", "-p", old_commit_hash, commit_hash], cwd=repo_path
            )
        except:
            return None  # mostly for the root commit

        return PerformanceCommit(
            commit_hash=commit_hash,
            subject=subject,
            message=message,
            date=date,
            files_changed=files_changed,
            diff_text=diff_text,
            repo_path=repo_path,
        )

    ######################### LLM-based Commit Filtering #########################

    @staticmethod
    def analysis_prompt(commit: PerformanceCommit):
        prompt = PERF_ANALYSIS_MESSAGE.format(
            diff_text=commit.diff_text, message=commit.message
        )

        if count_tokens(prompt) > MAX_COMMIT_TOKENS:
            diff_text = commit.diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
            prompt = PERF_ANALYSIS_MESSAGE.format(
                diff_text=diff_text, message=commit.message
            )

        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    @staticmethod
    def extract_tagged_section(response: str, tag: str) -> str:
        """Extract a tagged section, tolerating a missing or mismatched closing tag."""
        opening_match = re.search(rf"\[{re.escape(tag)}\]", response, re.IGNORECASE)
        if opening_match is None:
            raise ValueError(f"Missing [{tag}] tag")

        section_start = opening_match.end()
        next_tag = re.search(
            r"\[/?(?:REASON|ANSWER|APIS)\]",
            response[section_start:],
            re.IGNORECASE,
        )
        section_end = (
            section_start + next_tag.start() if next_tag is not None else len(response)
        )
        section = response[section_start:section_end].strip()
        if not section:
            raise ValueError(f"Empty [{tag}] section")
        return section

    @staticmethod
    def parse_json_dict(response: str):
        """Best-effort extraction of a JSON object from an LLM response.

        Tolerates Markdown code fences and stray text around the object.
        Returns the parsed dict, or None if no valid JSON object is found.
        """
        if not isinstance(response, str) or not response.strip():
            return None
        text = response.strip()

        # Strip a ```json ...``` fenced block if present (common with GLM).
        fence = re.search(
            r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if fence:
            text = fence.group(1).strip()

        # Fast path: the whole trimmed text is the object.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Fallback: scan for a brace-balanced object, skipping string contents.
        for start in range(len(text)):
            if text[start] != "{":
                continue
            depth = 0
            in_str = False
            escaped = False
            for end in range(start, len(text)):
                ch = text[end]
                if in_str:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break  # mismatched; try the next opening brace
                        break
        return None

    @staticmethod
    def parse_analysis_response(response: str) -> tuple[str, str]:
        # Prefer the structured JSON format. Fall back to the legacy
        # [REASON]/[ANSWER] tag format so old cached completions still parse.
        obj = PerfCommitAnalyzer.parse_json_dict(response)
        if obj is not None:
            reasoning = str(obj.get("reason", obj.get("reasoning", ""))).strip()
            answer_text = str(obj.get("answer", "")).strip().lower()
            answer_match = re.search(r"\b(yes|no)\b", answer_text)
            if reasoning and answer_match is not None:
                return reasoning, answer_match.group(1)

        reasoning = PerfCommitAnalyzer.extract_tagged_section(response, "REASON")
        answer_text = PerfCommitAnalyzer.extract_tagged_section(response, "ANSWER")
        answer_match = re.match(r"(YES|NO)\b", answer_text, re.IGNORECASE)
        if answer_match is None:
            raise ValueError("[ANSWER] section must begin with YES or NO")
        return reasoning, answer_match.group(1).lower()

    @staticmethod
    def llm_analysis(
        commits: list[PerformanceCommit], repo_path: Path, verbose: bool = False
    ):
        prompts = [PerfCommitAnalyzer.analysis_prompt(commit) for commit in commits]

        args = PerfCommitAnalyzer.build_llm_args(
            cache_stage="commit_filter", default_max_tokens=10000
        )

        responses = get_streaming_llm_completions(args, prompts)

        filtered = []
        for commit, completions in zip(commits, responses):
            if not completions or not isinstance(completions[0], str):
                print(f"Skipping {commit.commit_hash}: LLM returned no completion")
                continue

            try:
                reasoning, answer = PerfCommitAnalyzer.parse_analysis_response(
                    completions[0]
                )
            except ValueError as exc:
                print(f"Skipping {commit.commit_hash}: malformed LLM response ({exc})")
                continue

            if answer == "yes":
                commit.add_llm_reason(reasoning)
                filtered.append(commit)

            if verbose:
                print(f"Commit Hash: {commit.commit_hash}")
                print(f"Commit Message: {commit.message}")
                print(f"Reasoning: {reasoning}")
                print(f"Answer: {answer}")
                print("\n")

        # run retrieval to get affected files
        retriever = PerfCommitAnalyzer.retrieve_affected_files(filtered, repo_path)

        return filtered, retriever

    @staticmethod
    def retrieve_affected_files(commits: list[PerformanceCommit], repo_path: Path):
        from gso.collect.analysis.retriever import Retriever

        retriever = Retriever(repo_path)
        llm_args = PerfCommitAnalyzer.build_llm_args(
            cache_stage="affected_files", default_max_tokens=24000
        )
        retriever.retrieve_affected_files(commits, llm_args)
        return retriever

    ######################### LLM-based API Identification #########################

    @staticmethod
    def identify_api_prompt(commit: PerformanceCommit, retriever: Retriever):
        prompt = PERF_IDENTIFY_API_TASK.format(
            diff_text=commit.diff_text, message=commit.message
        )

        if count_tokens(prompt) > MAX_COMMIT_TOKENS:
            diff_text = commit.diff_text[:MAX_COMMIT_TOKENS] + "...(truncated)..."
            prompt = PERF_IDENTIFY_API_TASK.format(
                diff_text=diff_text, message=commit.message
            )

        tokens_so_far = count_tokens(prompt)

        file_content_prompt = "Some repo files:\n\n"
        for file_name in commit.affected_paths:
            content = retriever.file_content_map[file_name]
            new_content = f"File: {file_name}\n\n```{file_name.split('.')[-1]}\n{content}\n```\n\n"
            tokens_so_far += count_tokens(new_content)
            if tokens_so_far > MAX_OAI_TOKENS + THRESHOLD:
                new_content = (
                    new_content[: MAX_OAI_TOKENS - tokens_so_far - THRESHOLD]
                    + "...(truncated)...\n\n```"
                )
                file_content_prompt += new_content
                break
            file_content_prompt += new_content

        return [
            {
                "role": "system",
                "content": PERF_IDENTIFY_API_SYSTEM + "\n\n" + PERF_IDENTIFY_API_DOCS,
            },
            {"role": "user", "content": file_content_prompt},
            {
                "role": "user",
                "content": prompt,
            },
        ]

    @staticmethod
    def llm_get_apis(commits: list[PerformanceCommit], retriever: Retriever):
        if SKIP_API_ANALYSIS:
            for commit in commits:
                commit.add_apis(["SkippedAPIAnalysis"])
                commit.add_llm_api_reason(
                    "Skipped API analysis likely due to non python repo"
                )
            return

        prompts = [
            PerfCommitAnalyzer.identify_api_prompt(commit, retriever)
            for commit in commits
        ]

        args = PerfCommitAnalyzer.build_llm_args(
            cache_stage="api_identification", default_max_tokens=24000
        )

        responses = get_streaming_llm_completions(args, prompts)

        for commit, completions in zip(commits, responses):
            try:
                if not completions or not isinstance(completions[0], str):
                    raise ValueError("LLM returned no completion")
                reasoning = ""
                apis: list[str] = []
                obj = PerfCommitAnalyzer.parse_json_dict(completions[0])
                if obj is not None:
                    reasoning = str(obj.get("reason", obj.get("reasoning", ""))).strip()
                    apis_raw = obj.get("apis", [])
                    if isinstance(apis_raw, str):
                        apis = [a.strip() for a in apis_raw.split(",") if a.strip()]
                    elif isinstance(apis_raw, list):
                        apis = [str(a).strip() for a in apis_raw if str(a).strip()]
                if not reasoning:
                    # Fall back to the legacy [REASON]/[APIS] tag format.
                    reasoning = PerfCommitAnalyzer.extract_tagged_section(
                        completions[0], "REASON"
                    )
                    api_text = PerfCommitAnalyzer.extract_tagged_section(
                        completions[0], "APIS"
                    )
                    apis = [api.strip() for api in api_text.split(",") if api.strip()]
            except ValueError as exc:
                print(f"No APIs recorded for {commit.commit_hash}: {exc}")
                apis = []
                reasoning = "No APIs found"
            commit.add_apis(apis)
            commit.add_llm_api_reason(reasoning)

    ######################### Main Analysis #########################

    @staticmethod
    def get_performance_commits(
        repo_path: Path,
        no_grep: bool,
        max_year: int | None,
        max_commits: int | None = None,
        analyzed_commit_hashes: set[str] | None = None,
        analyzed_before: datetime | None = None,
    ) -> list[PerformanceCommit]:

        base_cmd = ["git", "log", "--pretty=format:%H", "-i"]
        grep_filters = [
            "--grep=perf",
            "--grep=performance",
            "--grep=optimize",
            "--grep=speed up",
            "--grep=speedup",
            "--grep=is slow",
            "--grep=faster",
            "--grep=overhead",
            "--grep=latency",
        ]

        # use grep to cut down commits to process
        if not no_grep:
            print("Using grep to filter commits")
            base_cmd = base_cmd[:3] + grep_filters + ["-i"]

        if analyzed_before is not None:
            base_cmd.append(f"--since={analyzed_before.isoformat()}")

        # get commit hashes, excluding commits covered by a previous analysis
        commit_hashes = run_git_command(base_cmd, cwd=repo_path).splitlines()
        if analyzed_commit_hashes is not None:
            commit_hashes = [
                commit_hash
                for commit_hash in commit_hashes
                if commit_hash not in analyzed_commit_hashes
            ]
            print("# New Candidate Commits:", len(commit_hashes))

        # Apply the limit after excluding previous results so it limits new work.
        if max_commits is not None:
            commit_hashes = commit_hashes[:max_commits]

        # Parse and process commits
        commits = []
        with Pool() as pool:
            commits = list(
                tqdm(
                    pool.starmap(
                        PerfCommitAnalyzer.process_commit,
                        [
                            (commit_hash, repo_path, max_year)
                            for commit_hash in commit_hashes
                        ],
                    ),
                    total=len(commit_hashes),
                )
            )

        commits = [commit for commit in commits if commit is not None]
        print("# Candidate Commits:", len(commits))
        if not commits:
            print("No new commits to analyze")
            return []

        # Proceed with LLM analysis without prompting. Non-interactive mode.
        print("Proceeding with LLM analysis (non-interactive mode)")

        # Record every commit sent for analysis, including commits rejected by the LLM.
        # This prevents rejected candidates from being analyzed again on the next run.
        if analyzed_commit_hashes is not None:
            analyzed_commit_hashes.update(commit.commit_hash for commit in commits)

        # LLM Analysis
        filtered, retriever = PerfCommitAnalyzer.llm_analysis(commits, repo_path)
        PerfCommitAnalyzer.llm_get_apis(filtered, retriever)
        print("# LLM Filtered Performance Commits:", len(filtered))

        # get diff stats for each performance commit
        for commit in tqdm(filtered, "Adding stats"):
            commit.add_stats(PerfCommitAnalyzer.parse_diff_for_stats(commit, repo_path))

        return filtered

    @staticmethod
    def analyze_repository(args) -> PerfAnalysis:
        repo_url = args.repo_url
        repo_owner, repo_name = repo_url.split("/")[-2:]
        repo_path = ANALYSIS_REPOS_DIR / repo_name
        output_file = ANALYSIS_COMMITS_DIR / f"{repo_name}_commits.json"
        ANALYSIS_REPOS_DIR.mkdir(parents=True, exist_ok=True)

        # Clone the repository if not already in ANALYSIS_DIR / "repos"
        if not os.path.exists(repo_path):
            subprocess.run(["git", "clone", repo_url, repo_path])

        existing_analysis = None
        analyzed_commit_hashes: set[str] = set()
        analyzed_before = None
        if output_file.exists():
            existing_analysis = PerfCommitAnalyzer.load_analysis(output_file)
            analyzed_commit_hashes.update(existing_analysis.analyzed_commit_hashes)
            analyzed_before = existing_analysis.analyzed_before
            if (
                analyzed_before is None
                and "analyzed_commit_hashes" not in existing_analysis.model_fields_set
            ):
                # Legacy files only stored accepted performance commits. Treat commits
                # dated before the artifact was written as part of that prior run.
                analyzed_before = datetime.fromtimestamp(
                    output_file.stat().st_mtime, tz=timezone.utc
                )
            # Backward compatibility for analysis files created before
            # analyzed_commit_hashes was recorded.
            analyzed_commit_hashes.update(
                commit.commit_hash for commit in existing_analysis.performance_commits
            )
            print(
                f"Reusing {len(existing_analysis.performance_commits)} existing "
                f"performance commits from {output_file}"
            )

        new_performance_commits = PerfCommitAnalyzer.get_performance_commits(
            repo_path,
            args.no_grep,
            args.max_year,
            getattr(args, "max_commits", None),
            analyzed_commit_hashes,
            analyzed_before,
        )

        existing_commits = (
            existing_analysis.performance_commits if existing_analysis else []
        )
        commits_by_hash = {
            commit.commit_hash: commit
            for commit in [*existing_commits, *new_performance_commits]
        }
        performance_commits = sorted(
            commits_by_hash.values(), key=lambda commit: commit.date, reverse=True
        )

        return PerfAnalysis(
            repo_url=repo_url,
            repo_owner=repo_owner,
            repo_name=repo_name,
            performance_commits=performance_commits,
            analyzed_commit_hashes=sorted(analyzed_commit_hashes),
            analyzed_before=analyzed_before,
        )

    @staticmethod
    def save_analysis(analysis: PerfAnalysis, output_file: Path):
        with open(output_file, "w") as f:
            f.write(analysis.model_dump_json(indent=2))

    @staticmethod
    def load_analysis(input_file: Path) -> PerfAnalysis:
        with open(input_file, "r") as f:
            data = json.load(f)
        return PerfAnalysis(**data)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch commits from a repository URL.")
    parser.add_argument("yaml_path", type=str, help="Path to the experiment YAML file.")
    parser.add_argument(
        "--max_year",
        type=int,
        required=False,
        default=None,
        help="Maximum year for commits",
    )
    parser.add_argument(
        "--no-grep",
        action="store_true",
        help="Disable grep-based commit filtering",
    )
    parser.add_argument(
        "--max-commits",
        "--max_commits",
        dest="max_commits",
        type=int,
        default=None,
        help="Maximum number of candidate commits to analyze",
    )
    args = parser.parse_args()
    configs = load_exp_config(args.yaml_path)
    PerfCommitAnalyzer.configure_llm(configs)

    if args.max_commits is None:
        args.max_commits = configs.get("max_commits")
    if args.max_commits is not None and (
        isinstance(args.max_commits, bool)
        or not isinstance(args.max_commits, int)
        or args.max_commits <= 0
    ):
        parser.error("max_commits must be a positive integer")

    args.repo_url = configs["repo_url"]
    if "api_docs" in configs:
        PERF_IDENTIFY_API_DOCS = configs["api_docs"]

    analysis = PerfCommitAnalyzer.analyze_repository(args)

    output_file = ANALYSIS_COMMITS_DIR / f"{analysis.repo_name}_commits.json"
    ANALYSIS_COMMITS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_APIS_DIR.mkdir(parents=True, exist_ok=True)
    PerfCommitAnalyzer.save_analysis(analysis, output_file)
