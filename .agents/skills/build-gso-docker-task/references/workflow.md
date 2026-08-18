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
# Real self-check: import the actual generation and execution entry points
# with PYTHONPATH unset so they resolve from the installed package instead
# of a shadowing src/ via PYTHONPATH. A bare `import gso` is not enough —
# this exercises every dependency the generation and execution entry points
# pull in (including diskcache).
env -u PYTHONPATH python -c "from gso.collect.generate.generate import *; from gso.collect.execute.execute import evaluate_generated_test"
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
- `stream` (defaults to `false`; set to `true` for streaming in analysis and generation)
- `extra_body` (optional mapping passed unchanged to the OpenAI-compatible request)
- `cache.commit_filter`, `cache.affected_files`, `cache.api_identification`, and `cache.test_generation`

`cache.test_generation` defaults to `false` so generation hits the live endpoint instead of reusing stale disk-cache entries; set it to `true` to reuse prior LLM completions.

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

### install_commands working-directory contract

The phase1 and phase2 scripts (`src/gso/collect/execute/phase1.txt`, `phase2.txt`) wrap every commit checkout in this sequence:

```bash
cd $repo_name && git stash -u   # /workspace -> /workspace/REPO_NAME
git checkout <commit>
$install_commands               # runs in /workspace/REPO_NAME
cd ..                           # must return to /workspace
for test_file in "<hash>"/test_*.py; do ...   # glob needs /workspace as cwd
```

`$install_commands` is interpolated verbatim (`"\n        ".join(...)` in `src/gso/collect/execute/skymgr.py`), so it runs as bare bash in the same shell. The contract: **after the install commands finish, the shell cwd must still be `/workspace/REPO_NAME`** so the following `cd ..` lands on `/workspace` and the `<hash>/test_*.py` glob matches `/workspace/<hash>/test_0.py`.

Breaking the contract produces a signature failure: the glob matches nothing, bash hands the literal pattern to `python`, and the container log shows `python: can't open file '//<hash>/test_*.py'` (the leading `//` means cwd was `/`). phase1 then emits no `working_pairs.json`, generation reports `collected 0 result(s)` for that slot, and the worker spends up to three semantic retries before skipping it — even though every test file was correctly written under `/workspace/<hash>/`.

Fixes and anti-patterns:

- **Installing from a subdirectory** (monorepo `packages/` layout, non-root `pyproject.toml`): isolate the `cd` in a subshell so the parent cwd is unchanged.
  ```bash
  # WRONG — leaks cwd into packages/markitdown
  cd packages/markitdown && uv pip install -e '.[all]'
  # RIGHT — subshell restores cwd
  ( cd packages/markitdown && uv pip install -e '.[all]' ) || uv pip install -e .
  ```
- **A stale trailing `cd /testbed` (or any absolute `cd`)** copied from another harness: remove it. The standard `Problem` install commands contain no `cd` at all.
- **A conditional `cd` on a file-layout check** (`if [ -f packages/.../pyproject.toml ]; then cd ...`): put each branch's `cd` inside a subshell, or reset cwd at the end of every branch. A `cd` on only the matched branch still breaks the next `cd ..`.

Verify locally before regenerating:

```bash
cd ~/buckets/gso_bucket/analysis/repos/REPO_NAME
# run the exact joined install_commands verbatim, then:
pwd
# must print the checkout root (.../REPO_NAME), never a subdirectory and never /
```

Because install commands are baked into `*_problems.json` and the repository image install step, changing them requires a regenerate with `--rebuild-docker-image` followed by a re-run of formal execution; leftover problem/result files still carry the broken commands. After editing the YAML, grep the saved `*_problems.json` to confirm the new commands propagated before execution.

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

The CLI is exposed through Fire. Keep generation's `--max_year` equal to the analysis cutoff; the workflow default is `2022`. Pass `--n 5` by default so each commit targets five accepted tests; override it only when the user explicitly requests another count. Other useful generation overrides include `--model_name`, `--multiprocess`, `--max_tokens`, `--openai_timeout`, `--stream`, and `--min_loc`. Docker overrides include `--docker-image`, `--docker-base-image`, `--docker-repo-path`, `--docker-cpus`, `--docker-memory`, `--docker-platform`, `--rebuild-docker-image`, `--keep-containers`, and `--keep-workspaces`.

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

### Diagnostic value of `--keep-workspaces`

With `--keep-workspaces` omitted (the default), each per-test validation
workspace is `shutil.rmtree`'d the moment its container exits. Only two
per-test artifacts survive: the container log under
`docker_logs/generation_validation/<cluster>.log` and the attempt record in
`EXP_ID_generation_retries_<timestamp>.json`. The host workspace that was
`docker cp`'d into the container — `phase1.sh`, `phase2.sh`,
`<pid>_task.yaml`, and one `<quick_hash>/test_i.py` per sampled test — is
gone, so from the log alone it is hard to tell a test that was never written
or failed to parse from a working-directory/glob mistake such as the
`python: can't open file '//<hash>/test_*.py'` signature. Listing the flag as
optional is correct for a healthy run (which keeps nothing extra); the
diagnostic value matters only while a `generation_validation` is failing.

Turn on `--keep-workspaces` (ideally scoped with `--api <failing_api>` and the
same `--n`/`--max_year` so you do not retain a workspace for every retry
across every commit) to retain the exact on-disk layout the container saw:

- Confirm `test_0.py` really landed under `<quick_hash>/`, i.e. the test was
  written and parseable rather than lost in generation/parsing.
- Read the expanded `phase1.sh` / `phase2.sh` with `install_commands`
  interpolated verbatim — the fastest way to spot a `cd` that leaked the cwd
  and broke the `cd .. ; <hash>/test_*.py` glob (see the install_commands
  working-directory contract in section 3).
- Reproduce the container layout locally without rebuilding the image.

A retained workspace is small: only phase scripts and per-commit `test_*.py`.
The repository itself lives in the Docker image, not the workspace, so
retention is cheap; reach for `--keep-containers` instead only when you need
the full container filesystem or an interactive `docker exec`. Each retained
workspace is announced on stdout as `Kept workspace for <pid> (commit(s)
<quick_hash>, run <n>): <path>`; correlate that path and the `test_0.py`
contents with the matching `raw_output`/`execution_log` entry in the retries
JSON and with `<cluster>.log` under `docker_logs/generation_validation/`.

Generated problems contain YAML `install_commands` when configured. If the field is absent, they contain the standard `Problem` commands. Verify the saved problems artifact contains the expected commands before execution.

### Per-iteration time budget and matrix scaling

Generation runs each proposed test through the same phase scripts as formal
execution: `phase1.txt` and `phase2.txt` hardcode `timeout 300s` around every
`python test_*.py` invocation, and `phase2.txt` repeats each reference and
candidate run **3 times** (`iterations=3`). Target tests are disabled
(`run_target_tests="false"` in `skymgr.py`), so each accepted slot costs **7
timed iterations** (1 phase1 reference + 3 phase2 reference + 3 phase2
candidate). Neither the timeout nor the iteration count is a CLI flag, so the
lever is the **workload size the LLM produces**, guided by the prompt and
`repo_instr`.

- Size each workload so the **base/parent commit's** `experiment` run lands
  around **1–60 s, ideally ~1–30 s**. A parent run near the 300 s ceiling
  (for example a ~220 s chart-rendering workload over hundreds of categories)
  leaves almost no margin for machine variance; one slow iteration trips the
  `timeout`, the slot is silently rejected as an execution failure, and the
  worker spends up to three LLM retries before skipping it. The generation
  prompt and the `repo_instr` template carry this 1–60 s target — set it
  explicitly in the YAML `repo_instr` for any repo whose natural workloads
  trend large (charts, large dataframes, model inference, image batches).
  Speedup is a ratio, not an absolute time, so a ~30 s parent run that the
  optimization halves is a clean 2× signal; you do **not** need a long base
  run to show a big speedup.
- Estimate total generation wall-time before launching. The matrix is `--n`
  × mapped commits × mapped APIs × ~7 timed iterations per slot, and each
  slot can spend up to **3 semantic retries** (a retry re-runs the slot).
  With `--n 5`, 3 commits, and 6 APIs that is 90 slots × ~7 iterations —
  well over an hour even at ~15 s per iteration, before retries. When the
  matrix is large, iterate cheaply with `--api package.target_api` first,
  lower `--n` to 2–3 for a broad sweep, and raise `--multiprocess` only as
  your LLM rate limit and Docker load allow.

### Running generation and execution in the background

Generation and formal execution are hour-scale and survive terminal disconnects
poorly when run through a foreground `tee` pipeline on an SSH PTY: a dropped
session can kill a 10-minute+ run mid-stream and waste the LLM spend and Docker
work already done. Decouple the process from the terminal and poll by tailing
the log instead of watching a live PTY:

```bash
# Generation: background, log to the workspace, survive shell exit
nohup python src/gso/collect/generate/generate.py "$workspace/experiment.yaml" \
    --n 5 --max_year 2022 \
    --docker-base-image gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
    --docker-repo-path ~/buckets/gso_bucket/analysis/repos/REPO_NAME \
    --docker-image gso-EXP_ID:latest \
    --docker-platform linux/amd64 \
    > "$workspace/logs/04-generate.log" 2>&1 &
echo $! > "$workspace/logs/04-generate.pid"
tail -f "$workspace/logs/04-generate.log"   # Ctrl-C stops only the tail
```

`setsid ... &` or a `tmux`/`screen` session are equivalent alternatives. The
same pattern applies to `execute.py` (section 5), which also repeats phase2 3×
and exposes `--poll-interval`; choose `--poll-interval 5` (or higher) for long
runs and background it. After an interruption, resume inspection with `tail`,
`grep`, or the retries/result JSON rather than re-running — an in-flight
generation that wrote `_problems.json` is not overwritten by a later failed
run, but a re-run still re-spends LLM calls for any slot not yet accepted.

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

Add `--api package.target_api` to run one API. Add `--rebuild-docker-image` to append Docker's `--no-cache`. Optional controls include `--docker-cpus`, `--docker-memory`, `--runs`, `--keep-containers`, `--keep-workspaces`, and `--poll-interval`. `--keep-workspaces` retains the per-task host workspace (phase scripts and `<quick_hash>/test_*.py`) for the same diagnostic reasons as generation (see section 4); the default deletes it on completion, leaving only the container log under `docker_logs/`. Formal execution repeats phase2 3× per slot under the same `timeout 300s`, so the 30–120 s base-run target and the background `nohup` + `tail -f` pattern from section 4 apply here too; pick a `--poll-interval` (e.g. `5`) that keeps log spam low over an hour-scale run.

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
