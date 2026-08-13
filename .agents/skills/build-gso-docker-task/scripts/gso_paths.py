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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve GSO paths and render commands without executing them."
    )
    parser.add_argument("yaml_path", type=Path, help="Experiment YAML")
    parser.add_argument("--repo-root", type=Path, required=True, help="GSO repo root")
    parser.add_argument("--api", help="Render generation/execution/evaluation for one API")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
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
    logs_path = exp_dir / "docker_logs"
    dataset_name = f"gso_{exp_id}"
    dataset_path = bucket / "datasets" / f"{dataset_name}_dataset.jsonl"
    repo_image = f"gso-{exp_id}:latest"
    pids_path = repo_root / f"{exp_id}_custom_pids.py"

    paths = {
        "repo_root": str(repo_root),
        "yaml_path": str(yaml_path),
        "exp_id": exp_id,
        "repo_name": repo_name,
        "repo_path": str(repo_path),
        "commits_path": str(commits_path),
        "apis_path": str(apis_path),
        "problems_path": str(problems_path),
        "docker_results_path": str(results_path),
        "docker_logs_path": str(logs_path),
        "custom_pids_path": str(pids_path),
        "dataset_path": str(dataset_path),
    }
    print(json.dumps(paths, indent=2, ensure_ascii=False))

    generate_lines: list[list[str | Path]] = [
        ["python", "src/gso/collect/generate/generate.py", yaml_path]
    ]
    if args.api:
        generate_lines.append(["--api", args.api])

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

    dataset_lines: list[list[str | Path]] = [
        ["python", "src/gso/collect/build_dataset.py"],
        ["--backend", "docker"],
        ["--exp_id", exp_id],
        ["--pids-file", pids_path],
        ["--dataset_name", dataset_name],
        ["--debug"],
    ]

    print("\n# Commands (run from repo_root)")
    print(command("python", "src/gso/collect/analysis/commits.py", yaml_path))
    print(command("python", "src/gso/collect/analysis/apis.py", repo_name))
    print(continued_command(generate_lines))
    print(command("docker", "image", "inspect", args.base_image))
    print(continued_command(execute_lines))
    print(continued_command(evaluate_lines))
    print(continued_command(dataset_lines))


if __name__ == "__main__":
    main()
