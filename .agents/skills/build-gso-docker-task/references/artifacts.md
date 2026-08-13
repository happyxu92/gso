# GSO artifact inspection

## Contents

1. Identity invariants
2. Safe inspection commands
3. Selection checks
4. Common stale-state traps

## 1. Identity invariants

- Operator-maintained task files use `${repo_root}/experiments/${repo_name}/`, where `repo_name` is derived from `repo_url`. New configurations, captured stage logs, plots, notes, and `custom_pids.py` belong there.
- Analysis artifacts use `repo_name`, derived from `repo_url`.
- GSO-generated experiment artifacts use YAML `exp_id` and remain under `~/buckets/gso_bucket/experiments/${exp_id}/`. The bucket directory is not the repository-scoped operator workspace.
- Problem IDs generally combine the repository name and API, but must be copied exactly from generated/evaluated results.
- Candidate selections use the exact 7-character value returned by `PerformanceCommit.quick_hash()`.
- Docker results use `EXP_ID_results_docker.json`; Sky results use `EXP_ID_results.json`. Never silently mix them.

## 2. Safe inspection commands

Use Python for structure-aware summaries rather than dumping large generated tests or possible log content to chat.

```bash
python - <<'PY' "$HOME/buckets/gso_bucket/analysis/commits/REPO_NAME_commits.json"
import json, sys
p = sys.argv[1]
d = json.load(open(p))
items = d.get("performance_commits", [])
print({"path": p, "repo_name": d.get("repo_name"), "commits": len(items)})
PY
```

```bash
python - <<'PY' "$HOME/buckets/gso_bucket/analysis/apis/REPO_NAME_ac_map.json"
import json, sys
p = sys.argv[1]
d = json.load(open(p))
m = d.get("api_to_commits", {})
print({"path": p, "repo_name": d.get("repo_name"), "apis": len(m)})
for api, commits in list(m.items())[:20]:
    print(api, len(commits))
PY
```

For Pydantic-serialized problem/result files, prefer importing GSO's loader from the repository environment:

```bash
python - <<'PY' "$HOME/buckets/gso_bucket/experiments/EXP_ID/EXP_ID_results_docker.json"
import sys
from gso.utils.io import load_problems
problems = load_problems(sys.argv[1])
print({"problems": len(problems), "valid": sum(p.is_valid() for p in problems)})
for p in problems:
    print(p.pid, p.api, len(p.commits), "valid=" + str(p.is_valid()))
PY
```

Validate the final JSONL and count rows:

```bash
python - <<'PY' "$HOME/buckets/gso_bucket/datasets/gso_EXP_ID_dataset.jsonl"
import json, sys
rows = 0
with open(sys.argv[1]) as f:
    for line_no, line in enumerate(f, 1):
        if line.strip():
            json.loads(line)
            rows += 1
print({"path": sys.argv[1], "rows": rows})
PY
```

## 3. Selection checks

Before writing `custom_pids.py`, confirm each pair against the Docker evaluation output and the loaded problems. Do not construct a PID from intuition about import paths. A valid pair can still yield no dataset row if its tests do not pass the dataset speedup thresholds.

Validate the custom file syntax and basic schema without executing arbitrary untrusted code:

```bash
python -m py_compile /absolute/path/to/gso/experiments/REPO_NAME/custom_pids.py
```

`build_dataset.py` loads the file with `runpy.run_path`, so only use a file created or trusted by the user. It must define a non-empty `TEST_PROBLEMS` dictionary. `LONG_RUNNING_PROBLEMS` defaults to an empty list when omitted, but define it explicitly for clarity.

## 4. Common stale-state traps

- `commits.py` reuses the analysis checkout and does not fetch it.
- A cached repository image may predate changes in the local checkout; rebuild it when the source snapshot must change.
- A single-API execution writes the common Docker results path. Inspect the file before assuming it still contains prior all-API results; use `--results-file` to preserve separate runs when needed.
- Generation failures preserve an existing problems file, so verify its modification time and contents before executing.
- Generation embeds `target_commit` and prefers YAML `install_commands`; when the field is absent it embeds the standard `Problem` commands. Review repository installation requirements and update the YAML before generation so the problems artifact is not stale.
- Captured stage output belongs in `experiments/REPO_NAME/logs/`; use timestamped or attempt-specific names when preserving retries.
- GSO's detailed Docker logs may persist under `~/buckets/gso_bucket/experiments/EXP_ID/docker_logs/`. Correlate those task directories and timestamps with both the current results file and the captured execution log.
