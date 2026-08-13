import json
import os
import shlex
from collections import defaultdict
from pathlib import Path


def zip_results(results_dir: Path):
    """Group base/commit/target/meta files by commit and test identifier."""
    file_groups = defaultdict(dict)

    for filename in results_dir.iterdir():
        if filename.suffix not in {".txt", ".json"}:
            continue
        parts = filename.stem.split("_", 2)
        if len(parts) >= 3:
            file_type, identifier = parts[0], parts[1] + "_" + parts[2]
            file_groups[identifier][file_type] = filename

    return file_groups


def collect_results(results_dir: Path) -> list[dict]:
    """Read a task results directory into the representation stored on Problem."""
    if not results_dir.exists():
        return []

    results = []
    for identifier, files in sorted(zip_results(results_dir).items()):
        try:
            commit, _ = identifier.split("_", 1)
        except ValueError:
            continue

        base_file = files.get("base")
        meta_file = files.get("meta")
        if base_file is None or meta_file is None:
            continue

        with meta_file.open() as f:
            meta = json.load(f)

        test_file = str(meta.get("test_file", ""))
        try:
            meta["test_id"] = int(Path(test_file).stem.rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            continue

        meta["commit"] = str(meta.get("commit", commit))
        meta["base_result"] = base_file.read_text()

        # Metadata initially contains output paths. Only retain result fields
        # when the corresponding file was actually produced.
        for result_type in ("commit", "target"):
            result_file = files.get(result_type)
            result_key = f"{result_type}_result"
            if result_file is None:
                meta.pop(result_key, None)
            else:
                meta[result_key] = result_file.read_text()

        results.append(meta)

    return results


def add_tokens_to_installs(install_commands: list[str]) -> list[str]:
    """Return install commands with optional credentials, without mutating input."""
    commands = list(install_commands)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        commands.append(f"export HF_TOKEN={shlex.quote(hf_token)}")
    return commands


def resolve_results_path(
    exp_dir: Path,
    exp_id: str,
    backend: str = "sky",
    results_file: str | Path | None = None,
) -> Path:
    """Resolve a results filename, relative to the experiment directory."""
    if results_file is None:
        suffix = "_results_docker.json" if backend == "docker" else "_results.json"
        return exp_dir / f"{exp_id}{suffix}"

    path = Path(results_file).expanduser()
    return path if path.is_absolute() else exp_dir / path
