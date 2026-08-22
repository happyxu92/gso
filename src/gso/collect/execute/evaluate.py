import argparse
import re
from pathlib import Path
from pprint import pformat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gso.collect.execute.helpers import resolve_results_path
from gso.constants import DATASET_DIR, EXPS_DIR
from gso.utils.io import load_problems


DEFAULT_MIN_SPEEDUP_FACTOR = 1.1
_TIME_PATTERN = re.compile(r"Execution time:\s+([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)s")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def has_non_python_changes(commit):
    """Check if commit includes changes to non-Python files."""
    ignore_list = ["py", "rst", "md", "txt", "yml", "toml", "gitignore"]
    return any(file.split(".")[-1] not in ignore_list for file in commit.files_changed)


def parse_times(time_str):
    if not isinstance(time_str, str):
        return []
    times = []
    for line in time_str.strip().splitlines():
        match = _TIME_PATTERN.fullmatch(line.strip())
        if match:
            times.append(float(match.group(1)))
    return times


def compute_stats(times):
    if not times:
        return None, None

    mean = float(np.mean(times))
    if not np.isfinite(mean):
        return None, None
    std_dev = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    return mean, std_dev


def print_prob_summary(prob):
    print("-" * 50)
    print("Problem.", prob.pid)
    print("  commits:", prob.num_commits())
    print("  tests:", prob.num_tests())
    print("  results:", prob.num_results())
    print("  valid_commits:", prob.num_valid_commits())
    print("  valid_tests:", prob.num_valid_tests())
    print("-" * 50)


def _result_stats(result: dict, key: str):
    return compute_stats(parse_times(result.get(key)))


def speedup_summary(
    prob,
    speedup_threshold=2,
    speedup_mode="target",
    non_python_only=False,
    python_only=False,
):
    """Analyze the latest run, skipping incomplete or malformed result pairs."""
    if speedup_mode not in {"target", "commit"}:
        raise ValueError("speedup_mode must be 'target' or 'commit'")
    if not prob.results:
        return {}, set(), set()

    _, latest_results = list(prob.results.items())[-1]
    result_stats = {}
    valid_commits, opt_commits = set(), set()

    for result in latest_results:
        test = result.get("test_file", "unknown-test")
        commit = next(
            (
                candidate
                for candidate in prob.commits
                if candidate.quick_hash() == result.get("commit")
            ),
            None,
        )
        if commit is None:
            continue
        if non_python_only and not has_non_python_changes(commit):
            continue
        if python_only and has_non_python_changes(commit):
            continue

        base_mean, base_std = _result_stats(result, "base_result")
        commit_mean, commit_std = _result_stats(result, "commit_result")
        target_mean, target_std = _result_stats(result, "target_result")

        if speedup_mode == "commit":
            after_mean, after_std = commit_mean, commit_std
        else:
            after_mean, after_std = target_mean, target_std

        # Local Docker collection normally compares the parent against the
        # candidate commit. Target results are optional and must not be
        # mistaken for the path placeholder stored in old metadata.
        if base_mean is None or after_mean is None or base_mean <= 0 or after_mean < 0:
            continue

        quick_hash = commit.quick_hash()
        valid_commits.add(quick_hash)
        opt_perc = ((base_mean - after_mean) / base_mean) * 100
        speedup_factor = float("inf") if after_mean == 0 else base_mean / after_mean
        loc_changed = commit.stats.get("num_non_test_edited_lines", 0)

        if base_mean > after_mean and opt_perc > speedup_threshold:
            test_id = result.get("test_id")
            if test_id is None:
                continue
            item = {
                "pid": prob.pid,
                "api": prob.api,
                "commit": result["commit"],
                "test_id": test_id,
                "base_mean": base_mean,
                "base_std": base_std or 0.0,
                "commit_mean": commit_mean,
                "commit_std": commit_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "after_mean": after_mean,
                "after_std": after_std or 0.0,
                "speedup_mode": speedup_mode,
                "opt_perc": opt_perc,
                "speedup_factor": speedup_factor,
                "loc_changed": loc_changed,
            }
            key = f"{prob.pid}-{test}"
            result_stats[key] = item
            opt_commits.add(quick_hash)

    return result_stats, valid_commits, opt_commits


def create_analysis_dataframe(problems) -> pd.DataFrame:
    """Convert analyzed problem data into a pandas DataFrame."""
    rows = []
    for problem_stats in problems.values():
        for key, stats in problem_stats.items():
            rows.append(
                {
                    "key": key,
                    "pid": stats["pid"],
                    "api": stats.get("api"),
                    "commit": stats["commit"],
                    "test_id": stats["test_id"],
                    "base_time": stats["base_mean"],
                    "commit_time": stats.get("commit_mean"),
                    "target_time": stats.get("target_mean"),
                    "after_time": stats["after_mean"],
                    "base_std": stats["base_std"],
                    "commit_std": stats.get("commit_std"),
                    "target_std": stats.get("target_std"),
                    "after_std": stats["after_std"],
                    "speedup_mode": stats["speedup_mode"],
                    "opt_perc": stats["opt_perc"],
                    "speedup_factor": stats["speedup_factor"],
                    "loc_changed": stats["loc_changed"],
                }
            )
    return pd.DataFrame(rows)


######### Plotting Functions #########


def plot_speedup_distribution(df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="opt_perc", bins=10, kde=True)
    plt.axvline(
        df["opt_perc"].mean(),
        color="r",
        linestyle="--",
        label=f'Mean: {df["opt_perc"].mean():.2f}%',
    )
    plt.title("Distribution of Performance Improvements")
    plt.xlabel("Opt (%)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(output_dir / "speedup_distribution.png")
    plt.close()


def plot_top_improvements(df: pd.DataFrame, output_dir: Path, top_n: int = 30):
    top_improvements = df.nlargest(top_n, "opt_perc")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_improvements, y="pid", x="opt_perc")
    plt.title(f"Top {top_n} Performance Improvements")
    plt.xlabel("Opt (%)")
    plt.ylabel("PID")
    plt.tight_layout()
    plt.savefig(output_dir / "top_improvements.png")
    plt.close()


def plot_execution_times_distribution(df: pd.DataFrame, output_dir: Path):
    mode = str(df["speedup_mode"].iloc[0]).title()
    plt.figure(figsize=(10, 6))
    plot_data = pd.DataFrame(
        {"Base Time": df["base_time"], f"{mode} Time": df["after_time"]}
    )
    sns.boxplot(data=plot_data)
    plt.title(f"Distribution of Base vs {mode} Execution Times")
    plt.ylabel("Execution Time (seconds)")
    plt.yscale("log")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "execution_times_distribution.png")
    plt.close()


def plot_top_pids_by_time(df: pd.DataFrame, output_dir: Path, top_n: int = 20):
    top_pids = df.groupby("pid")["base_time"].max().reset_index()
    top_pids = top_pids.nlargest(top_n, "base_time")
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=top_pids, y="pid", x="base_time", palette="viridis", hue="pid"
    )
    for i, value in enumerate(top_pids["base_time"]):
        ax.text(value + 0.1, i, f"{value:.2f}s", va="center")
    plt.title(f"Top {top_n} PIDs by Base Execution Time")
    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("PID")
    plt.tight_layout()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.savefig(output_dir / "top_time_pids.png")
    plt.close()
    return top_pids


######### Performance Summary #########


def create_performance_summary(df: pd.DataFrame) -> dict:
    return {
        "total_tests": len(df),
        "mean_speedup": df["opt_perc"].mean(),
        "median_speedup": df["opt_perc"].median(),
        "std_speedup": df["opt_perc"].std() if len(df) > 1 else 0.0,
        "max_speedup": df["opt_perc"].max(),
        "min_speedup": df["opt_perc"].min(),
    }


def select_pid_commits(
    dataframe: pd.DataFrame,
    min_speedup_factor: float = DEFAULT_MIN_SPEEDUP_FACTOR,
) -> list[tuple[str, str]]:
    """Select unique problem/commit pairs that can survive dataset filtering."""
    if min_speedup_factor <= 0:
        raise ValueError("min_speedup_factor must be positive")
    if dataframe.empty:
        return []

    qualifying = dataframe[dataframe["speedup_factor"] >= min_speedup_factor]
    return sorted(
        {
            (str(row.pid), str(row.commit))
            for row in qualifying[["pid", "commit"]].itertuples(index=False)
        }
    )


def write_pids_config(
    exp_id: str,
    pid_commits: list[tuple[str, str]],
    output_path: Path,
) -> Path:
    """Write a per-experiment PID selection without changing the shared pids.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_problems = {exp_id: pid_commits}
    contents = (
        '"""Auto-generated by gso.collect.execute.evaluate; do not edit manually."""\n\n'
        f"TEST_PROBLEMS = {pformat(test_problems, sort_dicts=False, width=100)}\n\n"
        "LONG_RUNNING_PROBLEMS = []\n"
    )
    output_path.write_text(contents, encoding="utf-8")
    return output_path


def build_evaluated_dataset(
    *,
    exp_id: str,
    dataframe: pd.DataFrame,
    backend: str,
    results_file: str | None,
    pids_output: str | None,
    dataset_name: str | None,
    min_speedup_factor: float,
) -> tuple[Path, Path] | None:
    """Export qualifying PIDs and build the corresponding JSONL dataset."""
    pid_commits = select_pid_commits(dataframe, min_speedup_factor)
    if not pid_commits:
        print(
            "No pid/commit pairs met the dataset threshold "
            f"({min_speedup_factor:.2f}x); skipping PID export and dataset build."
        )
        return None

    pids_path = (
        Path(pids_output).expanduser()
        if pids_output
        else EXPS_DIR / exp_id / f"{exp_id}_pids.py"
    )
    write_pids_config(exp_id, pid_commits, pids_path)
    print(f"Selected PID/commit pairs: {len(pid_commits)}")
    print(f"PIDs: {pids_path}")

    # Import lazily because build_dataset imports analysis helpers from this module.
    from gso.collect.build_dataset import main as build_dataset_main

    effective_dataset_name = dataset_name or f"gso_{exp_id}"
    build_dataset_main(
        exp_id=exp_id,
        push_to_hf=False,
        hf_username=None,
        dataset_name=effective_dataset_name,
        debug=False,
        backend=backend,
        results_file=results_file,
        pids_file=str(pids_path),
        min_speedup_factor=min_speedup_factor,
    )
    dataset_path = DATASET_DIR / f"{effective_dataset_name}_dataset.jsonl"
    if not dataset_path.is_file():
        raise RuntimeError(f"Dataset build completed without producing {dataset_path}")
    print(f"Dataset: {dataset_path}")
    return pids_path, dataset_path


def main(
    exp_id: str,
    specific_api: str | None = None,
    speedup_threshold: float = 2,
    loc_threshold: int | None = None,
    speedup_mode: str | None = "target",
    top_k: int = 10,
    non_python_only: bool = False,
    python_only: bool = False,
    top_by: str = "opt",
    backend: str = "sky",
    results_file: str | None = None,
    output_dir: str | None = None,
    build_dataset: bool = False,
    pids_output: str | None = None,
    dataset_name: str | None = None,
    min_speedup_factor: float = DEFAULT_MIN_SPEEDUP_FACTOR,
):
    """Analyze SkyPilot or local Docker collection results."""
    if backend not in {"sky", "docker"}:
        raise ValueError("backend must be 'sky' or 'docker'")
    if non_python_only and python_only:
        raise ValueError("--non-python-only and --python-only are mutually exclusive")
    if speedup_mode is None:
        speedup_mode = "commit" if backend == "docker" else "target"

    exp_dir = EXPS_DIR / exp_id
    results_path = resolve_results_path(
        exp_dir, exp_id, backend=backend, results_file=results_file
    )
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results not found: {results_path}. Run execute.py with --backend {backend}."
        )

    all_problems = load_problems(results_path)
    if specific_api:
        problems = [problem for problem in all_problems if problem.api == specific_api]
        if not problems:
            raise ValueError(f"No problem found for API: {specific_api}")
    else:
        problems = all_problems
    if not problems:
        raise ValueError(f"No problems found in {results_path}")

    plot_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else Path("plots") / exp_id / backend
    )

    num_with_results = 0
    analyzable_problems = 0
    opt_problems = {}
    err_problems = []
    opt_apis = set()
    all_commits = set()
    valid_commits_all = set()
    opt_commits_all = set()

    for problem in problems:
        all_commits.update(
            commit.quick_hash()
            for commit in problem.commits
            if commit.date.year >= 2016
        )
        if not problem.is_valid():
            err_problems.append(problem)
            continue

        num_with_results += 1
        stats, valid_commits, opt_commits = speedup_summary(
            problem,
            speedup_threshold=speedup_threshold,
            speedup_mode=speedup_mode,
            non_python_only=non_python_only,
            python_only=python_only,
        )
        if valid_commits:
            analyzable_problems += 1
        else:
            err_problems.append(problem)
        valid_commits_all.update(valid_commits)
        opt_commits_all.update(opt_commits)

        if stats:
            opt_problems[problem.pid] = stats
            opt_apis.update(value["api"] for value in stats.values())

    print("\n=== Performance Analysis Summary ===")
    print(f"Backend: {backend}")
    print(f"Results: {results_path}")
    print(f"Comparison: base -> {speedup_mode}")
    print(f"Total problems: {len(problems)}")
    print(f"Problems with result files: {num_with_results}")
    print(f"Analyzable problems: {analyzable_problems}")

    if not opt_problems:
        print("No optimization problems found.")
        if err_problems:
            print("Incomplete/errored APIs:")
            for problem in err_problems:
                print(f"  {problem.api}")
        return None, None

    plot_dir.mkdir(parents=True, exist_ok=True)
    dataframe = create_analysis_dataframe(opt_problems)
    plot_speedup_distribution(dataframe, plot_dir)
    plot_top_improvements(dataframe, plot_dir)
    plot_execution_times_distribution(dataframe, plot_dir)
    plot_top_pids_by_time(dataframe, plot_dir, top_n=20)
    summary = create_performance_summary(dataframe)

    optimized_pct = len(opt_problems) / max(analyzable_problems, 1) * 100
    print(f"Optimized problems: {len(opt_problems)} ({optimized_pct:.2f}%)")
    print(f"Optimized APIs: {opt_apis}")
    print(f"Plots: {plot_dir}")

    if err_problems:
        print("\nIncomplete/errored APIs:")
        for problem in err_problems:
            commits = ", ".join(
                f'{commit.quick_hash()} ({commit.date.strftime("%Y")})'
                for commit in problem.commits[:10]
            )
            print(f"  {problem.api}: {commits}")

    print("\nTest Analysis:")
    print(f"  Total tests analyzed: {summary['total_tests']}")
    print(f"  Mean Opt: {summary['mean_speedup']:.2f}%")
    print(f"  Median Opt: {summary['median_speedup']:.2f}%")
    print(f"  Standard deviation: {summary['std_speedup']:.2f}%")
    print(f"  Max Opt: {summary['max_speedup']:.2f}%")
    print(f"  Min Opt: {summary['min_speedup']:.2f}%")
    print(
        "\nSpeedup distribution:\n"
        f"{dataframe['opt_perc'].describe(percentiles=[0, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1])}"
    )
    print("=" * 35)

    print("\nCommit Analysis:")
    print(f"  Total commits: {len(all_commits)}")
    print(f"  Total valid commits: {len(valid_commits_all)}")
    print(f"  Total optimized commits: {len(opt_commits_all)}")

    best = (
        dataframe.groupby(["pid", "commit"])
        .agg({"opt_perc": "max", "speedup_factor": "max", "loc_changed": "first"})
        .reset_index()
    )

    if top_by == "opt":
        print("\nTop problems by Opt (best result per pid-commit):")
        for _, row in best.nlargest(top_k, "opt_perc").iterrows():
            print(
                f"  {row['pid']} ({row['commit']}): "
                f"{row['opt_perc']:.2f}% | {row['speedup_factor']:.2f}x"
            )

    if top_by == "loc":
        print("\nTop problems by LoC (best result per pid-commit):")
        for _, row in best.nlargest(top_k, "loc_changed").iterrows():
            print(f"  {row['pid']} ({row['commit']}): {row['loc_changed']}")

    if loc_threshold is not None and top_by == "loc_opt":
        print(f"\nTop commits by Opt > {speedup_threshold}% & LoC > {loc_threshold}:")
        top = best[best["loc_changed"] > loc_threshold]
        top_commits = top.nlargest(top_k, "loc_changed")["commit"].unique()
        for commit in top_commits:
            max_loc = top[top["commit"] == commit]["loc_changed"].max()
            print(f"  {commit} ({max_loc} lines):")
            for _, row in top[top["commit"] == commit].iterrows():
                print(
                    f"      {row['pid']} "
                    f"({row['opt_perc']:.2f}% | {row['speedup_factor']:.2f}x)"
                )

    if build_dataset:
        build_evaluated_dataset(
            exp_id=exp_id,
            dataframe=dataframe,
            backend=backend,
            results_file=results_file,
            pids_output=pids_output,
            dataset_name=dataset_name,
            min_speedup_factor=min_speedup_factor,
        )

    return dataframe, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze SkyPilot or local Docker performance results"
    )
    parser.add_argument("-e", "--exp_id", required=True, help="Experiment ID")
    parser.add_argument("-a", "--api", default=None, help="Specific API")
    parser.add_argument(
        "--backend", choices=["sky", "docker"], default="sky", help="Result backend"
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Results filename relative to the experiment directory, or absolute path",
    )
    parser.add_argument("--output-dir", default=None, help="Plot output directory")
    parser.add_argument(
        "-t",
        "--speedup_threshold",
        type=float,
        default=2,
        help="Minimum percentage improvement",
    )
    parser.add_argument(
        "-l", "--loc_threshold", type=int, default=None, help="Lines-of-code threshold"
    )
    parser.add_argument(
        "-m",
        "--speedup_mode",
        choices=["target", "commit"],
        default=None,
        help="Comparison mode (default: commit for Docker, target for SkyPilot)",
    )
    parser.add_argument("-k", "--top_k", type=int, default=10)
    parser.add_argument("--non-python-only", action="store_true")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--top_by", choices=["loc", "opt", "loc_opt"], default="opt")
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="export qualifying pid/commit pairs and build the dataset JSONL",
    )
    parser.add_argument(
        "--pids-output",
        default=None,
        help="PID config path (default: <bucket>/experiments/<exp_id>/<exp_id>_pids.py)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="dataset filename prefix (default: gso_<exp_id>)",
    )
    parser.add_argument(
        "--min-speedup-factor",
        type=_positive_float,
        default=DEFAULT_MIN_SPEEDUP_FACTOR,
        help=(
            "minimum speedup factor for PID selection "
            f"(default: {DEFAULT_MIN_SPEEDUP_FACTOR})"
        ),
    )
    args = parser.parse_args()

    if args.top_by == "loc_opt" and args.loc_threshold is None:
        raise ValueError("loc_threshold must be provided for top_by='loc_opt'")

    main(
        exp_id=args.exp_id,
        specific_api=args.api,
        speedup_threshold=args.speedup_threshold,
        loc_threshold=args.loc_threshold,
        speedup_mode=args.speedup_mode,
        top_k=args.top_k,
        non_python_only=args.non_python_only,
        python_only=args.python_only,
        top_by=args.top_by,
        backend=args.backend,
        results_file=args.results_file,
        output_dir=args.output_dir,
        build_dataset=args.build_dataset,
        pids_output=args.pids_output,
        dataset_name=args.dataset_name,
        min_speedup_factor=args.min_speedup_factor,
    )
