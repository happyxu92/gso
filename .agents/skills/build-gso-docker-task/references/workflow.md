# GSO local Docker workflow

## Contents

1. Repository workspace and preflight
2. Experiment YAML
3. Commit and API analysis
4. Docker-validated test generation
5. Docker execution
6. Evaluation and selection
7. Dataset build

All commands run from the GSO repository root. At the start of a task, derive `REPO_NAME` from the repository URL basename (removing an optional `.git`) and create a repository-scoped operator workspace:

```bash
workspace="$PWD/experiments/REPO_NAME"
mkdir -p "$workspace/logs" "$workspace/plots"
set -o pipefail
```

Store the experiment YAML, captured command output, plots, custom PID selection, and notes under this directory. Use stage-specific log names and add a timestamp or attempt suffix when prior logs must be retained. Every piped command below assumes `pipefail`, so a successful `tee` cannot hide a failed pipeline stage.

GSO-generated problems, results, and detailed Docker runtime logs remain under `~/buckets/gso_bucket`; this location is currently fixed in `src/gso/constants.py`. Do not confuse `experiments/REPO_NAME/` in the GSO checkout with `~/buckets/gso_bucket/experiments/EXP_ID/`.

## 1. Repository workspace and preflight

Create the workspace before writing configuration or starting analysis, then inspect prerequisites without exposing secret values:

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

For a new task, copy the skill template to `$workspace/experiment.yaml` and replace every placeholder. Do not create the YAML at the GSO repository root.

Required identity fields are `exp_id` and `repo_url`. Recommended local-Docker fields include `py_version`, `target_commit`, `install_commands`, and `repo_instr`. The `llm` mapping can include:

- `model_name`
- `base_url`
- `api_key_env`
- `multiprocess`
- `max_tokens`
- `openai_timeout`
- `cache.commit_filter`, `cache.affected_files`, and `cache.api_identification`

`api_docs` guides public API identification. Do not add a YAML `max_commits` field: the default single-repository workflow processes every matching commit after the year cutoff instead of truncating the candidate list.

The config's `llm` values apply to analysis and generation. Explicit generation CLI values take precedence. Never store the key itself in YAML. Generation prefers YAML `install_commands`; when the field is absent, `Problem` supplies its standard commands.

## 3. Commit and API analysis

```bash
python src/gso/collect/analysis/commits.py "$workspace/experiment.yaml" \
  --max_year 2022 \
  2>&1 | tee "$workspace/logs/01-commits.log"

python src/gso/collect/analysis/apis.py REPO_NAME \
  2>&1 | tee "$workspace/logs/02-apis.log"
```

Expected artifacts:

```text
~/buckets/gso_bucket/analysis/repos/REPO_NAME/
~/buckets/gso_bucket/analysis/commits/REPO_NAME_commits.json
~/buckets/gso_bucket/analysis/apis/REPO_NAME_ac_map.json
```

`commits.py` reuses an existing analysis checkout; it does not automatically update it. Its `--max_year 2022` behavior excludes commits dated in 2022 or earlier, so the analysis covers every matching commit after 2022. Do not pass `--max-commits` or `--max_commits`. `apis.py` takes the repository basename, not the experiment ID.

Before generation, inspect the analysis checkout's packaging and bootstrap files, such as `pyproject.toml`, `setup.py`, `setup.cfg`, requirements files, lockfiles, and contributor or CI installation instructions. Decide whether the current YAML commands install the repository and all runtime dependencies needed by generated tests. Typical reasons to customize them include required extras, a non-root package directory, native build prerequisites, editable-install limitations, or a repository-specific bootstrap command. If a change is needed, update the YAML first. If the YAML omits `install_commands` and the standard commands are sufficient, leave the field absent.

Generation now executes every proposed test in Docker, so complete Docker preflight before invoking the LLM:

```bash
docker image inspect gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  2>&1 | tee "$workspace/logs/03-docker-preflight.log"
test -d ~/buckets/gso_bucket/analysis/repos/REPO_NAME/.git
```

Confirm every mapped candidate and its parent exist in that checkout. Generation prepares `gso-EXP_ID:latest` once, using commit-only placeholders for image validation, before sending its first generation request.

## 4. Docker-validated test generation

Generate all mapped APIs:

```bash
python src/gso/collect/generate/generate.py "$workspace/experiment.yaml" \
  --n 5 \
  --max_year 2022 \
  --docker-base-image gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  --docker-repo-path ~/buckets/gso_bucket/analysis/repos/REPO_NAME \
  --docker-image gso-EXP_ID:latest \
  --docker-platform linux/amd64 \
  2>&1 | tee "$workspace/logs/04-generate.log"
```

Generate one exact API:

```bash
python src/gso/collect/generate/generate.py \
  "$workspace/experiment.yaml" \
  --api package.target_api \
  --n 5 \
  --max_year 2022 \
  --docker-base-image gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  --docker-repo-path ~/buckets/gso_bucket/analysis/repos/REPO_NAME \
  --docker-image gso-EXP_ID:latest \
  --docker-platform linux/amd64 \
  2>&1 | tee "$workspace/logs/04-generate-package.target_api.log"
```

Expected artifact:

```text
~/buckets/gso_bucket/experiments/EXP_ID/EXP_ID_problems.json
```

The CLI is exposed through Fire. Keep generation's `--max_year` equal to the analysis cutoff; the workflow default is `2022`. Pass `--n 5` by default so each commit targets five accepted tests; override it only when the user explicitly requests another count. Other useful generation overrides include `--model_name`, `--multiprocess`, `--max_tokens`, `--openai_timeout`, and `--min_loc`. Docker overrides include `--docker-image`, `--docker-base-image`, `--docker-repo-path`, `--docker-cpus`, `--docker-memory`, `--docker-platform`, `--rebuild-docker-image`, `--keep-containers`, and `--keep-workspaces`.

Generation behavior is sequential within each commit and parallel across commits:

1. Request exactly one response containing one JSON scenario block followed by one Python test block.
2. Parse and statically validate both, add the scenario to the test as comments, then execute the test against the candidate's parent and candidate in Docker.
3. Accept the scenario only when setup, reference execution, candidate execution, and equivalence checking produce usable results.
4. On format, static, or execution failure, feed the diagnostic back to the LLM and regenerate the complete pair from scratch, for up to three semantic retries.
5. If all three retries for that test slot fail, preserve its diagnostics, skip the slot, and proceed to the next slot. Do not abort the commit worker or the overall task merely because one slot failed.
6. Include only previously accepted scenarios in the next request so later tests differ materially. With the default `--n 5`, attempt all five slots even when an earlier slot is skipped; retain every accepted test, so the final count may be below five.

`--multiprocess` controls concurrent commit workers, while each worker attempts its `n` test slots one at a time. Account for LLM and Docker load when choosing it. Generation-time validation is a correctness gate, not a stable benchmark run.

Retry diagnostics are saved as `EXP_ID_generation_retries_<timestamp>.json`, including raw attempts, errors, scenarios, skipped slots, and available execution-log paths. A task-level terminal generation failure is saved as `EXP_ID_generation_failed_<timestamp>.json`; an existing problems file is not overwritten. Validation artifacts are written below:

```text
~/buckets/gso_bucket/experiments/EXP_ID/generation_validation/
~/buckets/gso_bucket/experiments/EXP_ID/docker_logs/generation_validation/
~/buckets/gso_bucket/experiments/EXP_ID/docker_logs/generation_validation/repository_image/build.log
```

Generated problems contain YAML `install_commands` when configured. If the field is absent, they contain the standard `Problem` commands. Verify the saved problems artifact contains the expected commands before execution.

## 5. Docker execution

Prefer the exact local analysis checkout:

```bash
docker image inspect gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  2>&1 | tee "$workspace/logs/05-docker-preflight.log"
test -d ~/buckets/gso_bucket/analysis/repos/REPO_NAME/.git

python src/gso/collect/execute/execute.py \
  --backend docker \
  --exp_id EXP_ID \
  --docker-base-image gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  --docker-repo-path ~/buckets/gso_bucket/analysis/repos/REPO_NAME \
  --docker-image gso-EXP_ID:latest \
  --docker-platform linux/amd64 \
  --machines 1 \
  2>&1 | tee "$workspace/logs/06-execute.log"
```

Add `--api package.target_api` to run one API. Add `--rebuild-docker-image` to append Docker's `--no-cache`. Optional controls include `--docker-cpus`, `--docker-memory`, `--runs`, `--keep-containers`, `--keep-workspaces`, and `--poll-interval`.

When `--docker-base-image` is supplied, execution builds the repository image. With `--docker-repo-path`, it copies the entire local checkout, including `.git`, into `/workspace/REPO_NAME`. Without `--docker-repo-path`, the repository image clones `repo_url` remotely. Omitting `--docker-base-image` requires `--docker-image` to exist already.

The runtime verifies required tools, the repo checkout, every candidate commit, and every candidate's parent before launching phase scripts. Run this formal execution even though generation already validated each test: it creates the Docker results used for evaluation and selection, and can repeat measurements under controlled conditions.

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
  --speedup_mode commit \
  --output-dir "$workspace/plots" \
  2>&1 | tee "$workspace/logs/07-evaluate.log"
```

Add `--api package.target_api` to narrow evaluation. Docker defaults to `commit` mode when `--speedup_mode` is omitted, but specify it for an auditable command.

The explicit `--output-dir` keeps plots under the repository workspace. Evaluation prints the top `(pid, 7-character commit)` pairs; its captured output stays in the workspace log. Select tasks based on stable repeated evidence, deterministic/equivalent results, reasonable duration, and meaningful parent-to-candidate speedup. Beware thermal throttling, background load, one-off outliers, I/O/network activity, and setup inside the timed region.

Create `$workspace/custom_pids.py` outside the shared source PID catalog:

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
  --pids-file "$workspace/custom_pids.py" \
  --dataset_name gso_EXP_ID \
  --debug \
  2>&1 | tee "$workspace/logs/08-build-dataset.log"
```

Expected artifact:

```text
~/buckets/gso_bucket/datasets/gso_EXP_ID_dataset.jsonl
```

The builder normally requires a per-test speedup factor of at least `1.2`; for groups with fewer than five strong tests it may supplement tests above `1.1`. It retains at most 20 tests per `(pid, commit)` unless `LONG_RUNNING_PROBLEMS` lowers the count. `--debug` identifies requested selections that did not survive these filters.
