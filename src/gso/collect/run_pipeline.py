#!/usr/bin/env python3
"""Run the GSO collection pipeline across a list of repositories.

For every repository in an input list this orchestrates the full GSO collection
workflow:

    1. commit analysis   -- src/gso/collect/analysis/commits.py
    2. API mapping        -- src/gso/collect/analysis/apis.py
    3. test generation    -- src/gso/collect/generate/generate.py (Docker-validated)
    4. test execution     -- src/gso/collect/execute/execute.py (local Docker backend)
    5. evaluation         -- src/gso/collect/execute/evaluate.py

Repositories run concurrently (``--concurrency``). Per-repository configuration
and run logs live under ``experiments/{repo}/`` in the GSO checkout, while
GSO-generated artifacts (clones, commit/API JSON, problems, results) stay under
the configurable GSO bucket (``~/buckets/gso_bucket`` by default).

Stages can be selected individually with ``--stages``. When generation is run on
its own it transparently reuses the commit/API artifacts already produced under
the bucket (``analysis/commits/{repo}_commits.json`` and
``analysis/apis/{repo}_ac_map.json``); it only needs those files to exist from a
prior ``commits,apis`` run.

Run from the GSO repository root with the GSO virtualenv interpreter so the
subprocesses can ``import gso``::

    .venv/bin/python src/gso/collect/run_pipeline.py \\
        assets/gso-python-performance-repositories

Common options::

    -j 3                       concurrent repositories (default 3)
    --stages all              one of: all | analysis | commits,apis,generate,...
    --buckets-dir ~/buckets   relocate the GSO bucket
    --max-year 2022           commit year cutoff (analysis + generation)
    -n 5                      target tests per commit
    --test-timeout 300        seconds allowed for each execute test invocation
    --api pkg.api             restrict generate/execute/evaluate to one API
    --api-key-envs KEY_1,KEY_2
                              rotate API key environment names across YAMLs
    --only repo1,repo2        restrict the repo list to these names
    --dry-run                 print planned commands without executing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Repo-relative locations
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent  # src/gso/collect
_REPO_ROOT = _THIS_DIR.parents[2]  # GSO repository root
_DEFAULT_TEMPLATE = _REPO_ROOT / "assets" / "experiment.yaml"

DEFAULT_BUCKETS_DIR = Path.home() / "buckets"
DEFAULT_BASE_IMAGE = "gso-base:ubuntu22.04-py312-uv0.12.0-amd64"
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
DEFAULT_MAX_YEAR = 2022
DEFAULT_MAX_COMMITS = 300
DEFAULT_TESTS_PER_COMMIT = 5
DEFAULT_EXEC_MACHINES = 1
DEFAULT_TEST_TIMEOUT = 300
DEFAULT_CONCURRENCY = 3
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")

# Canonical pipeline order. Filtering preserves this order.
STAGES = ("commits", "apis", "generate", "execute", "evaluate")
STAGE_ALIASES = {
    "all": set(STAGES),
    "analysis": {"commits", "apis"},
}

# Shared log filename per stage (kept aligned with the skill's workflow logs).
STAGE_LOG_NAMES = {
    "commits": "01-commits.log",
    "apis": "02-apis.log",
    "generate": "04-generate.log",
    "execute": "06-execute.log",
    "evaluate": "07-evaluate.log",
}

_PRINT_LOCK = threading.Lock()
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_API_KEY_ENV_LINE_RE = re.compile(
    r"^(?P<prefix>[ \t]*api_key_env[ \t]*:[ \t]*)"
    r"[^#\r\n]*?(?P<comment>[ \t]+#.*)?$",
    re.MULTILINE,
)
_GENERATION_PROGRESS_PREFIX = "Generation progress:"
_GENERATION_EVENT_PREFIX = "Generation event:"
_GENERATION_OUTPUT_PREFIXES = (
    _GENERATION_PROGRESS_PREFIX,
    _GENERATION_EVENT_PREFIX,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def timestamped(msg: str) -> str:
    """Prefix a message with the current Beijing time in ISO 8601 format."""
    timestamp = datetime.now(BEIJING_TIMEZONE).isoformat(timespec="seconds")
    return f"[{timestamp}] {msg}"


def log(msg: str) -> None:
    """Thread-safe, timestamped console print."""
    with _PRINT_LOCK:
        print(timestamped(msg), flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Run the GSO collection pipeline across many repositories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "repo_list",
        type=Path,
        help="CSV repository list (columns: repository,url,stars)",
    )
    p.add_argument(
        "-j",
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"concurrent repositories (default: {DEFAULT_CONCURRENCY})",
    )
    p.add_argument(
        "--buckets-dir",
        type=Path,
        default=DEFAULT_BUCKETS_DIR,
        help=(
            "parent directory of the GSO bucket (default: ~/buckets). "
            "The bucket itself is {buckets-dir}/gso_bucket and can also be "
            "set directly with the GSO_BUCKET_DIR environment variable."
        ),
    )
    p.add_argument(
        "--template",
        type=Path,
        default=_DEFAULT_TEMPLATE,
        help="experiment YAML template to render per repository",
    )
    p.add_argument(
        "--experiments-root",
        type=Path,
        default=_REPO_ROOT / "experiments",
        help="root for per-repo workspaces (config + logs), default: {repo_root}/experiments",
    )
    p.add_argument(
        "--stages",
        default="all",
        help=(
            "comma-separated subset of: "
            + ", ".join(STAGES)
            + " (or aliases: all, analysis). Example: --stages commits,apis"
        ),
    )
    p.add_argument(
        "--max-year",
        type=int,
        default=DEFAULT_MAX_YEAR,
        help="commit year cutoff (default: 2022)",
    )
    p.add_argument(
        "--no-grep",
        action="store_true",
        help="disable grep-based commit filtering in commits.py",
    )
    p.add_argument(
        "--max-commits",
        type=int,
        default=DEFAULT_MAX_COMMITS,
        help=(
            "cap the newest candidate commits analyzed "
            f"(default: {DEFAULT_MAX_COMMITS})"
        ),
    )
    p.add_argument(
        "-n",
        "--tests-per-commit",
        type=int,
        default=DEFAULT_TESTS_PER_COMMIT,
        help="target tests per commit for generation (default: 5)",
    )
    p.add_argument(
        "--api", default=None, help="restrict generate/execute/evaluate to one API"
    )
    p.add_argument(
        "--api-key-env",
        "--api-key-envs",
        dest="api_key_envs",
        action="append",
        default=[],
        metavar="ENV[,ENV...]",
        help=(
            "API key environment variable names to rotate across generated YAML "
            "configs; comma-separated and/or repeatable (keys stay out of YAML)"
        ),
    )
    p.add_argument(
        "--base-image",
        default=DEFAULT_BASE_IMAGE,
        help="local Docker base image (must be built beforehand)",
    )
    p.add_argument(
        "--docker-platform",
        default=DEFAULT_DOCKER_PLATFORM,
        help="Docker platform, e.g. linux/amd64",
    )
    p.add_argument(
        "--docker-cpus", type=float, default=None, help="CPU limit per container"
    )
    p.add_argument(
        "--docker-memory", default=None, help="memory limit per container, e.g. 8g"
    )
    p.add_argument(
        "--exec-machines",
        type=int,
        default=DEFAULT_EXEC_MACHINES,
        help="concurrent Docker containers for execute.py (default: 1)",
    )
    p.add_argument(
        "--rebuild-docker-image",
        action="store_true",
        help="rebuild the repository image without Docker cache",
    )
    p.add_argument(
        "--keep-containers", action="store_true", help="keep containers after execution"
    )
    p.add_argument(
        "--keep-workspaces", action="store_true", help="keep generated host workspaces"
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="execute.py status polling interval in seconds",
    )
    p.add_argument(
        "--test-timeout",
        type=_positive_int,
        default=DEFAULT_TEST_TIMEOUT,
        help=(
            "timeout in seconds for each test invocation during execute "
            f"(default: {DEFAULT_TEST_TIMEOUT})"
        ),
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter for GSO subprocesses (default: current python)",
    )
    p.add_argument(
        "--only",
        default=None,
        help="restrict to a comma-separated subset of repository names",
    )
    p.add_argument(
        "--overwrite-config",
        action="store_true",
        help="overwrite an existing per-repo experiment.yaml",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="echo subprocess output to the console (noisy with concurrency)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned commands without executing or writing configs",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers: paths, repo list, config rendering
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BucketPaths:
    gso_bucket: Path
    analysis_repos: Path
    analysis_commits: Path
    analysis_apis: Path
    experiments: Path


def compute_paths(buckets_dir: Path) -> BucketPaths:
    gso_bucket = buckets_dir / "gso_bucket"
    return BucketPaths(
        gso_bucket=gso_bucket,
        analysis_repos=gso_bucket / "analysis" / "repos",
        analysis_commits=gso_bucket / "analysis" / "commits",
        analysis_apis=gso_bucket / "analysis" / "apis",
        experiments=gso_bucket / "experiments",
    )


def repo_name_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    name = Path(path).name
    if name.endswith(".git"):
        name = name[:-4]
    if not name or name in {".", ".."}:
        raise ValueError(f"cannot derive repository name from repo_url: {url!r}")
    return name


def read_repo_list(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "url" not in reader.fieldnames:
            raise SystemExit(
                f"repo list {path} must be a CSV with a 'url' column "
                f"(found: {reader.fieldnames})"
            )
        for row in reader:
            repository = (row.get("repository") or "").strip()
            url = (row.get("url") or "").strip()
            if not repository or not url:
                continue
            rows.append((repository, url))
    return rows


def normalize_api_key_envs(raw_values: list[str] | None) -> list[str]:
    """Parse and validate repeatable/comma-separated environment names."""
    names: list[str] = []
    for raw_value in raw_values or []:
        for raw_name in raw_value.split(","):
            name = raw_name.strip()
            if not name:
                raise SystemExit("--api-key-envs contains an empty environment name")
            if not _ENV_VAR_NAME_RE.fullmatch(name):
                raise SystemExit(f"invalid API key environment variable name: {name!r}")
            if name in names:
                raise SystemExit(f"duplicate API key environment variable name: {name}")
            names.append(name)
    return names


def set_config_api_key_env(config_text: str, api_key_env: str) -> str:
    """Set the sole api_key_env entry while preserving the rest of the YAML."""
    matches = list(_API_KEY_ENV_LINE_RE.finditer(config_text))
    if len(matches) != 1:
        raise ValueError(
            "experiment YAML must contain exactly one 'api_key_env:' entry "
            f"when --api-key-envs is used (found {len(matches)})"
        )

    def replacement(match: re.Match[str]) -> str:
        comment = match.group("comment") or ""
        return f'{match.group("prefix")}{json.dumps(api_key_env)}{comment}'

    return _API_KEY_ENV_LINE_RE.sub(replacement, config_text, count=1)


def render_config(
    template_text: str,
    exp_id: str,
    url: str,
    api_key_env: str | None = None,
) -> str:
    text = template_text
    text = text.replace("REPLACE_EXP_ID", exp_id)
    text = text.replace("https://github.com/REPLACE_OWNER/REPLACE_REPOSITORY", url)
    if api_key_env is not None:
        text = set_config_api_key_env(text, api_key_env)
    # Backstop: if the template uses bare owner/repo placeholders elsewhere.
    return text


def assign_api_key_envs(
    rows: list[tuple[str, str]], api_key_envs: list[str]
) -> list[tuple[tuple[str, str], str | None]]:
    """Assign environment names round-robin in the input repository order."""
    if not api_key_envs:
        return [(row, None) for row in rows]
    return [
        (row, api_key_envs[index % len(api_key_envs)]) for index, row in enumerate(rows)
    ]


def resolve_stages(stages_arg: str) -> set[str]:
    key = stages_arg.strip().lower()
    if key in STAGE_ALIASES:
        return set(STAGE_ALIASES[key])
    expanded: set[str] = set()
    for part in stages_arg.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in STAGE_ALIASES:
            expanded |= STAGE_ALIASES[token]
        elif token in STAGES:
            expanded.add(token)
        else:
            raise SystemExit(
                f"unknown stage '{token}'; valid: {list(STAGES)} "
                f"or aliases: {sorted(STAGE_ALIASES)}"
            )
    return expanded


# ---------------------------------------------------------------------------
# Minimal .env loader so preflight token checks are meaningful in the parent.
# Subprocesses load .env themselves via gso/__init__.py; this is best-effort.
# ---------------------------------------------------------------------------
def _load_dotenv_parent() -> Path | None:
    configured = os.getenv("GSO_ENV_FILE")
    if configured:
        env_path = Path(configured).expanduser()
        if env_path.is_file():
            _apply_dotenv(env_path)
            return env_path
        return None
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            _apply_dotenv(candidate)
            return candidate
    return None


def _apply_dotenv(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Per-repo result / workspace
# ---------------------------------------------------------------------------
@dataclass
class RepoResult:
    repo: str
    exp_id: str
    url: str
    workspace: Path
    stages_run: list[str] = field(default_factory=list)
    stages_ok: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    elapsed: float = 0.0


def prepare_workspace(
    repo: str,
    url: str,
    repository: str,
    workspace: Path,
    logs_dir: Path,
    plots_dir: Path,
    template_text: str,
    overwrite_config: bool,
    dry_run: bool,
    api_key_env: str | None = None,
) -> Path:
    if not dry_run:
        workspace.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
    config_path = workspace / f"{repo}.yaml"
    if dry_run:
        return config_path
    existed = config_path.exists()
    if existed and not overwrite_config:
        if api_key_env is None:
            log(f"[{repo}] reuse existing config: {config_path}")
        else:
            existing_text = config_path.read_text(encoding="utf-8")
            updated = set_config_api_key_env(existing_text, api_key_env)
            config_path.write_text(updated, encoding="utf-8")
            log(
                f"[{repo}] updated api_key_env={api_key_env} in existing config: "
                f"{config_path}"
            )
    else:
        rendered = render_config(template_text, repo, url, api_key_env)
        config_path.write_text(rendered, encoding="utf-8")
        action = "overwrote" if existed else "wrote"
        log(f"[{repo}] {action} config: {config_path}")
    return config_path


# ---------------------------------------------------------------------------
# Stage command construction
# ---------------------------------------------------------------------------
def _script_path(*parts: str) -> str:
    return str(_REPO_ROOT.joinpath("src", "gso", "collect", *parts))


def _docker_common_args(
    repo_checkout: Path,
    repo_image: str,
    args: argparse.Namespace,
) -> list[str]:
    extras: list[str] = []
    if repo_checkout and repo_checkout.is_dir():
        extras += ["--docker-repo-path", str(repo_checkout)]
    if args.docker_cpus is not None:
        extras += ["--docker-cpus", str(args.docker_cpus)]
    if args.docker_memory:
        extras += ["--docker-memory", args.docker_memory]
    if args.rebuild_docker_image:
        extras.append("--rebuild-docker-image")
    if args.keep_containers:
        extras.append("--keep-containers")
    if args.keep_workspaces:
        extras.append("--keep-workspaces")
    # docker-base-image / docker-image / docker-platform are stage-specific.
    return extras


def build_stages(
    repo: str,
    exp_id: str,
    repo_checkout: Path,
    config_path: Path,
    plots_dir: Path,
    args: argparse.Namespace,
    paths: BucketPaths,
) -> list[tuple[str, list[str], Path]]:
    """Return [(stage, command, log_path)] in canonical pipeline order."""
    py = args.python
    repo_image = f"gso-{exp_id.lower()}:latest"
    commits_json = paths.analysis_commits / f"{repo}_commits.json"
    apis_json = paths.analysis_apis / f"{repo}_ac_map.json"
    problems_json = paths.experiments / exp_id / f"{exp_id}_problems.json"
    results_json = paths.experiments / exp_id / f"{exp_id}_results_docker.json"

    docker_common = _docker_common_args(repo_checkout, repo_image, args)
    docker_image_args = [
        "--docker-base-image",
        args.base_image,
        "--docker-image",
        repo_image,
        "--docker-platform",
        args.docker_platform,
    ]

    stages: list[tuple[str, list[str], Path]] = []

    # 1. commits
    cmd = [
        py,
        _script_path("analysis", "commits.py"),
        str(config_path),
        "--max_year",
        str(args.max_year),
    ]
    if args.no_grep:
        cmd.append("--no-grep")
    if args.max_commits is not None:
        cmd += ["--max-commits", str(args.max_commits)]
    stages.append(("commits", cmd, commits_json))

    # 2. apis (takes the repo basename, not exp_id)
    stages.append(("apis", [py, _script_path("analysis", "apis.py"), repo], apis_json))

    # 3. generate (Fire CLI; positional yaml_path)
    cmd = [
        py,
        _script_path("generate", "generate.py"),
        str(config_path),
        "--n",
        str(args.tests_per_commit),
        "--max_year",
        str(args.max_year),
    ]
    if args.api:
        cmd += ["--api", args.api]
    cmd += docker_image_args + docker_common
    stages.append(("generate", cmd, problems_json))

    # 4. execute (argparse; local Docker backend)
    cmd = [
        py,
        _script_path("execute", "execute.py"),
        "--backend",
        "docker",
        "--exp_id",
        exp_id,
    ]
    if args.api:
        cmd += ["--api", args.api]
    cmd += docker_image_args + ["--machines", str(args.exec_machines)]
    cmd += ["--test-timeout", str(args.test_timeout)]
    cmd += docker_common
    if args.poll_interval is not None:
        cmd += ["--poll-interval", str(args.poll_interval)]
    stages.append(("execute", cmd, results_json))

    # 5. evaluate
    cmd = [
        py,
        _script_path("execute", "evaluate.py"),
        "--backend",
        "docker",
        "--exp_id",
        exp_id,
        "--speedup_mode",
        "commit",
        "--output-dir",
        str(plots_dir),
        "--build-dataset",
    ]
    if args.api:
        cmd += ["--api", args.api]
    stages.append(("evaluate", cmd, None))

    return stages


# ---------------------------------------------------------------------------
# Subprocess execution with per-repo log capture
# ---------------------------------------------------------------------------
def run_command(
    command: list[str],
    log_path: Path,
    repo_label: str,
    verbose: bool,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "$ " + " ".join(shlex.quote(str(token)) for token in command)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(timestamped(header) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if line == "":
                break
            # tqdm redraws a single terminal row with carriage returns. Turn
            # those redraws into ordinary lines so every captured update can
            # carry its own timestamp in the persistent stage log.
            normalized_line = line.replace("\r\n", "\n").replace("\r", "\n")
            output_lines = normalized_line.splitlines()
            if not output_lines:
                output_lines = [""]
            for output_line in output_lines:
                logf.write(timestamped(output_line) + "\n")
                if verbose:
                    log(f"[{repo_label}] {output_line}")
                else:
                    # Keep normal mode quiet except for the structured progress
                    # and lifecycle records added by generate.py.
                    matching_prefixes = [
                        prefix
                        for prefix in _GENERATION_OUTPUT_PREFIXES
                        if prefix in output_line
                    ]
                    if matching_prefixes:
                        marker_index = max(
                            output_line.rfind(prefix) for prefix in matching_prefixes
                        )
                        generation_output = output_line[marker_index:]
                        log(f"[{repo_label}] {generation_output}")
            logf.flush()
        proc.wait()
    return proc.returncode, time.time() - started


def count_perf_commits(commits_path: Path) -> int | None:
    try:
        data = json.loads(commits_path.read_text(encoding="utf-8"))
        return len(data.get("performance_commits", []))
    except Exception:
        return None


def stage_prerequisites(repo: str, exp_id: str, paths: BucketPaths) -> dict:
    """Map each stage to (producer_stage, artifact_path, hint).

    A stage selected on its own (i.e. its producer is NOT in the current
    ``--stages``) is expected to reuse the producer's artifact from the bucket.
    If that artifact is missing the stage cannot run and must be skipped
    instead of letting the GSO subprocess crash late with a FileNotFoundError.
    ``commits`` has no producer (it clones the repository itself) and is absent.
    """
    return {
        "apis": (
            "commits",
            paths.analysis_commits / f"{repo}_commits.json",
            "run --stages commits first",
        ),
        "generate": (
            "apis",
            paths.analysis_apis / f"{repo}_ac_map.json",
            "run --stages commits,apis first",
        ),
        "execute": (
            "generate",
            paths.experiments / exp_id / f"{exp_id}_problems.json",
            "run --stages generate first",
        ),
        "evaluate": (
            "execute",
            paths.experiments / exp_id / f"{exp_id}_results_docker.json",
            "run --stages execute first",
        ),
    }


# ---------------------------------------------------------------------------
# Per-repo driver
# ---------------------------------------------------------------------------
def run_repo(
    repo_row: tuple[str, str],
    api_key_env: str | None,
    args: argparse.Namespace,
    paths: BucketPaths,
    template_text: str,
    selected: set[str],
) -> RepoResult:
    repository, url = repo_row
    repo = repo_name_from_url(url)
    exp_id = repo  # convention: exp_id == repository name
    workspace = args.experiments_root / repo
    logs_dir = workspace / "logs"
    plots_dir = workspace / "plots"
    result = RepoResult(repo=repo, exp_id=exp_id, url=url, workspace=workspace)
    started = time.time()

    try:
        if api_key_env is not None:
            log(f"[{repo}] assigned api_key_env={api_key_env}")
        config_path = prepare_workspace(
            repo,
            url,
            repository,
            workspace,
            logs_dir,
            plots_dir,
            template_text,
            args.overwrite_config,
            args.dry_run,
            api_key_env,
        )
        repo_checkout = paths.analysis_repos / repo
        stages = build_stages(
            repo, exp_id, repo_checkout, config_path, plots_dir, args, paths
        )
        stages = [
            (name, cmd, artifact)
            for (name, cmd, artifact) in stages
            if name in selected
        ]

        # When a stage is selected but the stage that *produces* its input is
        # not, we expect to reuse that input from the bucket. Verify it is
        # actually there; if it is missing, skip this stage and everything
        # downstream instead of letting the subprocess crash with a
        # FileNotFoundError deep in the GSO code.
        prereq = stage_prerequisites(repo, exp_id, paths)
        planned: list[tuple[str, list[str], Path | None]] = []
        skip_downstream = False
        for name, command, artifact in stages:
            if skip_downstream:
                result.stages_run.append(name)
                log(f"[{repo}] !! {name} skipped: upstream prerequisite missing")
                continue
            producer, prereq_artifact, hint = prereq.get(name, (None, None, None))
            if (
                producer is not None
                and producer not in selected
                and prereq_artifact is not None
                and not prereq_artifact.exists()
            ):
                log(
                    f"[{repo}] !! {name} skipped: prerequisite '{producer}' "
                    f"artifact not found: {prereq_artifact}"
                )
                log(f"[{repo}]    hint: {hint}")
                result.stages_run.append(name)
                result.skipped = True
                result.skip_reason = (
                    f"{name} missing {producer} artifact " f"({prereq_artifact.name})"
                )
                skip_downstream = True
                continue
            planned.append((name, command, artifact))

        # Local analysis checkout is preferred for the Docker build but is not
        # strictly required (generate.py/execute.py fall back to a remote
        # clone). Warn only when a Docker stage actually will run, so an
        # all-skipped repo does not emit a misleading remote-clone warning.
        will_run_docker = any(name in {"generate", "execute"} for name, _, _ in planned)
        if will_run_docker and not repo_checkout.is_dir():
            log(
                f"[{repo}] WARNING: local analysis checkout {repo_checkout} "
                f"not found; generate/execute will fall back to a remote "
                f"repository clone (slower, may not reuse the exact analyzed "
                f"commits)."
            )

        if args.dry_run:
            for name, cmd, artifact in planned:
                rendered = " ".join(shlex.quote(str(token)) for token in cmd)
                log(f"[{repo}] DRY {name}: {rendered}  (artifact: {artifact or '-'})")
            if not planned:
                log(f"[{repo}] DRY: nothing would run (prerequisites missing)")
            result.skipped = True
            result.skip_reason = "dry-run"
            return result

        if not planned:
            log(f"[{repo}] no runnable stages after prerequisite checks")
            return result

        for name, command, artifact in planned:
            result.stages_run.append(name)
            log_path = logs_dir / STAGE_LOG_NAMES[name]
            rendered = " ".join(shlex.quote(str(token)) for token in command)
            log(f"[{repo}] >>> {name}  (log: {log_path})")
            log(f"[{repo}]     {rendered}")
            rc, elapsed = run_command(command, log_path, repo, args.verbose)
            if rc != 0:
                result.failed_stage = name
                result.error = f"{name} exited with code {rc}; see {log_path}"
                log(f"[{repo}] <<< {name}: FAILED rc={rc} ({elapsed:.1f}s)")
                break
            log(f"[{repo}] <<< {name}: ok ({elapsed:.1f}s)")

            if artifact is not None and not artifact.exists():
                result.failed_stage = name
                result.error = f"{name} exited cleanly but did not produce {artifact}"
                log(f"[{repo}] !!! {result.error}")
                break

            # Smart skip: no performance commits means nothing downstream to do.
            if name == "commits":
                count = count_perf_commits(artifact)
                if count is not None and count == 0:
                    log(
                        f"[{repo}] commits analysis found 0 performance "
                        f"commits; skipping remaining stages."
                    )
                    result.stages_ok.append(name)
                    result.skipped = True
                    result.skip_reason = "no performance commits"
                    break

            result.stages_ok.append(name)
    except Exception as exc:  # noqa: BLE001 - surface any runner failure
        result.error = f"{type(exc).__name__}: {exc}"
        log(f"[{repo}] !!! runner exception: {result.error}")
    finally:
        result.elapsed = time.time() - started
    return result


# ---------------------------------------------------------------------------
# Preflight + summary
# ---------------------------------------------------------------------------
def preflight(args: argparse.Namespace) -> None:
    if not _REPO_ROOT.joinpath("src", "gso").is_dir():
        log(f"WARNING: does not look like a GSO repo root: {_REPO_ROOT}")
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.Architecture}}"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            log("WARNING: docker daemon not reachable; docker stages will fail")
        else:
            log(f"docker ok (arch={proc.stdout.strip()})")
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: docker preflight failed: {exc}")

    if args.base_image:
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", args.base_image],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode != 0:
                log(
                    f"WARNING: base image '{args.base_image}' not found; "
                    f"build it before running generate/execute."
                )
            else:
                log(f"base image ok: {args.base_image}")
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: base image preflight failed: {exc}")

    env_path = _load_dotenv_parent()
    if env_path:
        log(f"loaded .env from {env_path}")
    # Check LLM key names without printing values.
    key_candidates = args.api_key_envs or [
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_KEY",
    ]
    present = [name for name in key_candidates if os.getenv(name)]
    missing = [name for name in key_candidates if not os.getenv(name)]
    if present:
        log(f"credentials present in env: {', '.join(present)}")
        if args.api_key_envs and missing:
            log(
                "WARNING: configured API key environment variable(s) not set: "
                + ", ".join(missing)
            )
    else:
        log(
            "WARNING: none of the API key environment variables are set: "
            + ", ".join(key_candidates)
            + "; put them in a .env (or export them) before analysis/generation."
        )


def classify(result: RepoResult) -> str:
    if result.skipped and result.skip_reason == "dry-run":
        return "DRY"
    if result.skipped:
        return f"SKIP({result.skip_reason})"
    if result.failed_stage:
        return f"FAIL@{result.failed_stage}"
    if result.error:
        return "FAIL"
    if not result.stages_ok:
        return "NOOP"
    return "OK"


def print_summary(results: list[RepoResult]) -> None:
    log("=" * 78)
    log("GSO pipeline summary")
    log("=" * 78)
    for result in sorted(results, key=lambda r: r.repo):
        status = classify(result)
        ok = ",".join(result.stages_ok) or "-"
        detail = result.error or ""
        log(
            f"{result.repo:28s} {status:24s} ok=[{ok}] "
            f"{result.elapsed:6.1f}s  {detail}"
        )
    log("=" * 78)
    total = len(results)
    by_status: dict[str, int] = {}
    for result in results:
        by_status[classify(result)] = by_status.get(classify(result), 0) + 1
    log(
        f"{total} repo(s): "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    args.api_key_envs = normalize_api_key_envs(args.api_key_envs)

    args.buckets_dir = args.buckets_dir.expanduser().resolve()
    args.experiments_root = args.experiments_root.expanduser().resolve()
    args.template = args.template.expanduser().resolve()
    paths = compute_paths(args.buckets_dir)
    # Relay the bucket location to the GSO subprocesses (constants.py honors it).
    os.environ["GSO_BUCKET_DIR"] = str(paths.gso_bucket)

    if not args.repo_list.is_file():
        raise SystemExit(f"repository list not found: {args.repo_list}")
    if not args.template.is_file():
        raise SystemExit(f"experiment template not found: {args.template}")

    template_text = "" if args.dry_run else args.template.read_text(encoding="utf-8")

    rows = read_repo_list(args.repo_list)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        rows = [row for row in rows if repo_name_from_url(row[1]) in wanted]
    if not rows:
        raise SystemExit("no repositories to process")

    selected = resolve_stages(args.stages)

    log(
        f"GSO pipeline: {len(rows)} repo(s), concurrency={args.concurrency}, "
        f"stages={args.stages} -> {sorted(selected)}"
    )
    log(f"bucket: {paths.gso_bucket}")
    log(f"workspaces: {args.experiments_root}/<repo>/{{logs,plots}}")
    log(f"interpreter: {args.python}")
    if args.api_key_envs:
        log("API key env rotation: " + " -> ".join(args.api_key_envs))
    if not args.dry_run:
        preflight(args)

    results: list[RepoResult] = []
    assignments = assign_api_key_envs(rows, args.api_key_envs)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {
            pool.submit(
                run_repo,
                row,
                api_key_env,
                args,
                paths,
                template_text,
                selected,
            ): row
            for row, api_key_env in assignments
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                repo = repo_name_from_url(row[1])
                result = RepoResult(
                    repo=repo,
                    exp_id=repo,
                    url=row[1],
                    workspace=args.experiments_root / repo,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            log(
                f"[{result.repo}] ===== {classify(result)} ===== "
                f"{result.error or ''}"
            )

    print_summary(results)

    failures = [
        result
        for result in results
        if result.failed_stage is not None
        or (result.error is not None and not result.skipped)
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
