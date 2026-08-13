# GSO local Docker workflow

## Contents

1. Preflight
2. Experiment YAML
3. Commit and API analysis
4. Test generation
5. Docker execution
6. Evaluation and selection
7. Dataset build

All commands run from the GSO repository root. The implementation stores data under `~/buckets/gso_bucket`; this location is currently fixed in `src/gso/constants.py`.

## 1. Preflight

Inspect without exposing secret values:

```bash
test -d src/gso
python --version
python -c 'import gso; print(gso.__file__)'
docker info --format '{{.Architecture}}'
docker image inspect gso-base:ubuntu22.04-py312-uv0.5.4-amd64
```

Confirm the configured key variable is present using a boolean-only check:

```bash
python -c 'import os; raise SystemExit(0 if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY is not set")'
```

If the YAML uses another `llm.api_key_env`, substitute that name. GSO loads `.env` from the current directory or a parent; `GSO_ENV_FILE` selects a non-default file. Common optional credentials are `GHAPI_TOKEN` and `HF_TOKEN`.

## 2. Experiment YAML

Required identity fields are `exp_id` and `repo_url`. Recommended local-Docker fields include `py_version`, `target_commit`, `install_commands`, and `repo_instr`. The `llm` mapping can include:

- `model_name`
- `base_url`
- `api_key_env`
- `multiprocess`
- `max_tokens`
- `openai_timeout`
- `cache.commit_filter`, `cache.affected_files`, and `cache.api_identification`

`api_docs` guides public API identification. `max_commits` is an optional positive top-level integer limiting the newest commit candidates.

The config's `llm` values apply to analysis and generation. Explicit generation CLI values take precedence. Never store the key itself in YAML. Generation prefers YAML `install_commands`; when the field is absent, `Problem` supplies its standard commands.

## 3. Commit and API analysis

```bash
python src/gso/collect/analysis/commits.py /absolute/path/to/experiment.yaml \
  --max_year 2021

python src/gso/collect/analysis/apis.py REPO_NAME
```

Expected artifacts:

```text
~/buckets/gso_bucket/analysis/repos/REPO_NAME/
~/buckets/gso_bucket/analysis/commits/REPO_NAME_commits.json
~/buckets/gso_bucket/analysis/apis/REPO_NAME_ac_map.json
```

`commits.py` reuses an existing analysis checkout; it does not automatically update it. `apis.py` takes the repository basename, not the experiment ID.

Before generation, inspect the analysis checkout's packaging and bootstrap files, such as `pyproject.toml`, `setup.py`, `setup.cfg`, requirements files, lockfiles, and contributor or CI installation instructions. Decide whether the current YAML commands install the repository and all runtime dependencies needed by generated tests. Typical reasons to customize them include required extras, a non-root package directory, native build prerequisites, editable-install limitations, or a repository-specific bootstrap command. If a change is needed, update the YAML first. If the YAML omits `install_commands` and the standard commands are sufficient, leave the field absent.

## 4. Test generation

Generate all mapped APIs:

```bash
python src/gso/collect/generate/generate.py /absolute/path/to/experiment.yaml
```

Generate one exact API:

```bash
python src/gso/collect/generate/generate.py \
  /absolute/path/to/experiment.yaml \
  --api package.target_api
```

Expected artifact:

```text
~/buckets/gso_bucket/experiments/EXP_ID/EXP_ID_problems.json
```

The CLI is exposed through Fire. Useful overrides include `--model_name`, `--multiprocess`, `--n`, `--max_tokens`, `--openai_timeout`, `--max_year`, and `--min_loc`. A generation failure is saved as `EXP_ID_generation_failed_<timestamp>.json` and does not overwrite the existing problems file.

Generated problems contain YAML `install_commands` when configured. If the field is absent, they contain the standard `Problem` commands. Verify the saved problems artifact contains the expected commands before execution.

## 5. Docker execution

Prefer the exact local analysis checkout:

```bash
docker image inspect gso-base:ubuntu22.04-py312-uv0.5.4-amd64
test -d ~/buckets/gso_bucket/analysis/repos/REPO_NAME/.git

python src/gso/collect/execute/execute.py \
  --backend docker \
  --exp_id EXP_ID \
  --docker-base-image gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  --docker-repo-path ~/buckets/gso_bucket/analysis/repos/REPO_NAME \
  --docker-image gso-EXP_ID:latest \
  --docker-platform linux/amd64 \
  --machines 1
```

Add `--api package.target_api` to run one API. Add `--rebuild-docker-image` to append Docker's `--no-cache`. Optional controls include `--docker-cpus`, `--docker-memory`, `--runs`, `--keep-containers`, `--keep-workspaces`, and `--poll-interval`.

When `--docker-base-image` is supplied, execution builds the repository image. With `--docker-repo-path`, it copies the entire local checkout, including `.git`, into `/workspace/REPO_NAME`. Without `--docker-repo-path`, the repository image clones `repo_url` remotely. Omitting `--docker-base-image` requires `--docker-image` to exist already.

The runtime verifies required tools, the repo checkout, every candidate commit, and every candidate's parent before launching phase scripts.

Expected artifacts:

```text
~/buckets/gso_bucket/experiments/EXP_ID/EXP_ID_results_docker.json
~/buckets/gso_bucket/experiments/EXP_ID/docker_logs/
~/buckets/gso_bucket/experiments/EXP_ID/docker_logs/repository_image/build.log
```

## 6. Evaluation and selection

```bash
python src/gso/collect/execute/evaluate.py \
  --backend docker \
  --exp_id EXP_ID \
  --speedup_mode commit
```

Add `--api package.target_api` to narrow evaluation. Docker defaults to `commit` mode when `--speedup_mode` is omitted, but specify it for an auditable command.

Evaluation writes plots under `plots/EXP_ID/docker` by default and prints the top `(pid, 7-character commit)` pairs. Select tasks based on stable repeated evidence, deterministic/equivalent results, reasonable duration, and meaningful parent-to-candidate speedup. Beware thermal throttling, background load, one-off outliers, I/O/network activity, and setup inside the timed region.

Create `custom_pids.py` outside the shared source PID catalog:

```python
TEST_PROBLEMS = {
    "EXP_ID": [
        ("PID", "abcdef1"),
    ],
}

LONG_RUNNING_PROBLEMS = []
```

Entries in `LONG_RUNNING_PROBLEMS` use `(pid, commit, max_test_count)`.

## 7. Dataset build

```bash
python src/gso/collect/build_dataset.py \
  --backend docker \
  --exp_id EXP_ID \
  --pids-file /absolute/path/to/custom_pids.py \
  --dataset_name gso_EXP_ID \
  --debug
```

Expected artifact:

```text
~/buckets/gso_bucket/datasets/gso_EXP_ID_dataset.jsonl
```

The builder normally requires a per-test speedup factor of at least `1.2`; for groups with fewer than five strong tests it may supplement tests above `1.1`. It retains at most 20 tests per `(pid, commit)` unless `LONG_RUNNING_PROBLEMS` lowers the count. `--debug` identifies requested selections that did not survive these filters.
