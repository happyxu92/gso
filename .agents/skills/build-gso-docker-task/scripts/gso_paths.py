#!/usr/bin/env python3
"""Resolve GSO identities/artifacts and render local Docker workflow commands."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from urllib.parse import urlparse

import yaml


DEFAULT_BASE_IMAGE = "gso-base:ubuntu22.04-py312-uv0.5.4-amd64"
DEFAULT_MAX_YEAR = 2022


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def repository_name(repo_url: str) -> str:
    path = urlparse(repo_url).path.rstrip("/")
    name = Path(path).name
    if name.endswith(".git"):
        name = name[:-4]
    if not name or name in {".", ".."}:
        raise ValueError(f"Cannot derive repository name from repo_url: {repo_url!r}")
    return name


def command(*parts: str | Path) -> str:
    return " ".join(quote(part) for part in parts)


def continued_command(lines: list[list[str | Path]]) -> str:
    separator = " " + "\\" + "\n  "
    return separator.join(" ".join(quote(part) for part in line) for line in lines)


def logged_command(rendered_command: str, log_path: Path) -> str:
    return f"{rendered_command} 2>&1 | tee {quote(log_path)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve GSO paths and render commands without executing them."
    )
    parser.add_argument("yaml_path", type=Path, help="Experiment YAML")
    parser.add_argument("--repo-root", type=Path, required=True, help="GSO repo root")
    parser.add_argument(
        "--api", help="Render generation/execution/evaluation for one API"
    )
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument(
        "--max-year",
        type=int,
        default=DEFAULT_MAX_YEAR,
        help="Commit year cutoff passed to analysis and generation (default: 2022)",
    )
    parser.add_argument("--docker-platform", default="linux/amd64")
    parser.add_argument("--rebuild-docker-image", action="store_true")
    args = parser.parse_args()

    yaml_path = args.yaml_path.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    if not yaml_path.is_file():
        parser.error(f"experiment YAML not found: {yaml_path}")
    if not (repo_root / "src" / "gso").is_dir():
        parser.error(f"not a GSO repository root: {repo_root}")

    with yaml_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        parser.error("experiment YAML must contain a mapping")
    exp_id = config.get("exp_id")
    repo_url = config.get("repo_url")
    if not isinstance(exp_id, str) or not exp_id.strip():
        parser.error("experiment YAML requires a non-empty string exp_id")
    if not isinstance(repo_url, str) or not repo_url.strip():
        parser.error("experiment YAML requires a non-empty string repo_url")
    exp_id = exp_id.strip()
    repo_name = repository_name(repo_url.strip())

    bucket = Path.home() / "buckets" / "gso_bucket"
    repo_path = bucket / "analysis" / "repos" / repo_name
    commits_path = bucket / "analysis" / "commits" / f"{repo_name}_commits.json"
    apis_path = bucket / "analysis" / "apis" / f"{repo_name}_ac_map.json"
    exp_dir = bucket / "experiments" / exp_id
    problems_path = exp_dir / f"{exp_id}_problems.json"
    results_path = exp_dir / f"{exp_id}_results_docker.json"
    docker_logs_path = exp_dir / "docker_logs"
    generation_results_path = exp_dir / "generation_validation"
    generation_logs_path = docker_logs_path / "generation_validation"
    dataset_name = f"gso_{exp_id}"
    dataset_path = bucket / "datasets" / f"{dataset_name}_dataset.jsonl"
    repo_image = f"gso-{exp_id}:latest"
    workspace_dir = repo_root / "experiments" / repo_name
    run_logs_path = workspace_dir / "logs"
    plots_path = workspace_dir / "plots"
    pids_path = workspace_dir / "custom_pids.py"

    paths = {
        "repo_root": str(repo_root),
        "workspace_dir": str(workspace_dir),
        "yaml_path": str(yaml_path),
        "exp_id": exp_id,
        "repo_name": repo_name,
        "repo_path": str(repo_path),
        "commits_path": str(commits_path),
        "apis_path": str(apis_path),
        "problems_path": str(problems_path),
        "generation_results_path": str(generation_results_path),
        "generation_logs_path": str(generation_logs_path),
        "docker_results_path": str(results_path),
        "docker_logs_path": str(docker_logs_path),
        "run_logs_path": str(run_logs_path),
        "plots_path": str(plots_path),
        "custom_pids_path": str(pids_path),
        "dataset_path": str(dataset_path),
    }
    print(json.dumps(paths, indent=2, ensure_ascii=False))

    generate_lines: list[list[str | Path]] = [
        ["python", "src/gso/collect/generate/generate.py", yaml_path],
        ["--max_year", str(args.max_year)],
    ]
    if args.api:
        generate_lines.append(["--api", args.api])
    generate_lines.extend(
        [
            ["--docker-base-image", args.base_image],
            ["--docker-repo-path", repo_path],
            ["--docker-image", repo_image],
            ["--docker-platform", args.docker_platform],
        ]
    )
    if args.rebuild_docker_image:
        generate_lines.append(["--rebuild-docker-image"])

    execute_lines: list[list[str | Path]] = [
        ["python", "src/gso/collect/execute/execute.py"],
        ["--backend", "docker"],
        ["--exp_id", exp_id],
    ]
    if args.api:
        execute_lines.append(["--api", args.api])
    execute_lines.extend(
        [
            ["--docker-base-image", args.base_image],
            ["--docker-repo-path", repo_path],
            ["--docker-image", repo_image],
            ["--docker-platform", args.docker_platform],
        ]
    )
    if args.rebuild_docker_image:
        execute_lines.append(["--rebuild-docker-image"])
    execute_lines.append(["--machines", "1"])

    evaluate_lines: list[list[str | Path]] = [
        ["python", "src/gso/collect/execute/evaluate.py"],
        ["--backend", "docker"],
        ["--exp_id", exp_id],
    ]
    if args.api:
        evaluate_lines.append(["--api", args.api])
    evaluate_lines.append(["--speedup_mode", "commit"])
    evaluate_lines.append(["--output-dir", plots_path])

    dataset_lines: list[list[str | Path]] = [
        ["python", "src/gso/collect/build_dataset.py"],
        ["--backend", "docker"],
        ["--exp_id", exp_id],
        ["--pids-file", pids_path],
        ["--dataset_name", dataset_name],
        ["--debug"],
    ]

    print("\n# Commands (run from repo_root in a shell supporting pipefail)")
    print(command("mkdir", "-p", run_logs_path, plots_path))
    print("set -o pipefail")
    print(
        logged_command(
            command(
                "python",
                "src/gso/collect/analysis/commits.py",
                yaml_path,
                "--max_year",
                str(args.max_year),
            ),
            run_logs_path / "01-commits.log",
        )
    )
    print(
        logged_command(
            command("python", "src/gso/collect/analysis/apis.py", repo_name),
            run_logs_path / "02-apis.log",
        )
    )
    print(
        logged_command(
            command("docker", "image", "inspect", args.base_image),
            run_logs_path / "03-docker-preflight.log",
        )
    )
    print(
        logged_command(
            continued_command(generate_lines), run_logs_path / "04-generate.log"
        )
    )
    print(
        logged_command(
            continued_command(execute_lines), run_logs_path / "06-execute.log"
        )
    )
    print(
        logged_command(
            continued_command(evaluate_lines), run_logs_path / "07-evaluate.log"
        )
    )
    print(
        logged_command(
            continued_command(dataset_lines),
            run_logs_path / "08-build-dataset.log",
        )
    )


if __name__ == "__main__":
    main()
