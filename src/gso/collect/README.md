# Collection Framework


## Overview

1. **Commit Extraction & Filtering**: Extracts potential performance-related commits from a given repository using LLMs.
2. **API Identification**: Use RAG w/ LLM pipeline to identify affected high-level APIs for each performance commit.
3. **Performance Test Generation**: Generates performance tests for the identified API-Commit pairs using LLMs.
4. **Test Execution**: Execute performane tests and identify problems (API-Commit pairs) that show performance improvements.

**Prerequisite**:

Install [r2e](https://github.com/r2e-project/r2e) for llm helpers and parallel sampling:
```
uv pip install git+https://github.com/r2e-project/r2e
```

Setup [Github](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens), [OpenAI](https://platform.openai.com/api-keys), [HuggingFace](https://huggingface.co/docs/hub/en/security-tokens) tokens
```
export GHAPI_TOKEN="github_token"
export OPENAI_API_KEY="openai_key"
export HF_TOKEN="huggingface_token"
```

Collection commands also load these values from a `.env` file in the current
directory or one of its parents (existing process environment variables take
precedence):
```dotenv
GHAPI_TOKEN="github_token"
OPENAI_API_KEY="openai_key"
HF_TOKEN="huggingface_token"
```

For a non-default file, set `GSO_ENV_FILE=/path/to/file.env` before running the
pipeline. The variable named by `llm.api_key_env` can also be stored in this
file.


## Usage

### 1. Configure experiments

First pick an experiment ID, usually the repository name (say `repo`) -- you will use this ID to refer to the experiment in following steps. Experiments can be configured using a simple YAML file with the following structure:

```yaml
exp_id: "repo"
repo_url: "https://github.com/username/repo"
max_commits: 100  # optional; newest matching candidate commits to analyze
llm:
    model_name: "custom-model"
    base_url: "https://openai-compatible.example/v1"
    api_key_env: "OPENAI_API_KEY"
    multiprocess: 4
    max_tokens: 32768
    openai_timeout: 600
    stream: false
    extra_body:
        enable_thinking: false
    cache:
        commit_filter: true
        affected_files: true
        api_identification: true
py_version: 3.9
target_commit: "main"
install_commands: []
```

You can add the repository URL and custom python version & installation commands. When `generate.py` sees an empty or omitted `install_commands`, it asks the local Codex CLI to inspect the local repository in a read-only sandbox and infer a repository-specific command list. A successful result is written back to both the input YAML and its experiment copy. If Codex is unavailable or inference fails, generation continues with the code-defined defaults. A non-empty list skips inference and is used unchanged. The optional `llm` mapping configures the model used by commit analysis and performance-test generation; explicit generation CLI values for `model_name`, `multiprocess`, `max_tokens`, `openai_timeout`, `stream`, and `extra_body` take precedence. Both analysis and generation default to non-stream responses; set `llm.stream: true` to enable streaming for all four LLM stages. Provider-specific request fields can be supplied through the generic `llm.extra_body` mapping and are passed unchanged to both transport modes. Generated performance tests default to `max_tokens: 32768` and `openai_timeout: 600` seconds. Transport failures are retried at most five times. The optional `llm.cache` mapping controls response caching for the `commit_filter`, `affected_files`, and `api_identification` analysis stages. All three stages default to `true`. Each value must be a YAML boolean. You can also specify `api_docs` and `repo_instr` (free form strings) to specify APIs to focus on during analysis and custom performance test generation tips. See examples in the [experiments/](/exps/) directory.



### 2. Commit Analysis Pipeline

The [analysis/](/src/gso/collect/analysis/) directory contains the performance commit analysis pipeline. It identifies and analyzes performance-related commits in Python repositories and then maps them to high-level APIs that are affected by the changes. More details in the [readme](/src/gso/collect/analysis/README.md).

Run the pipeline on any repository using the `commits.py` script:
```bash
python src/gso/collect/analysis/commits.py /path/to/experiment.yaml
python src/gso/collect/analysis/apis.py repo
```

The commit analysis results are saved as a JSON file in `ANALYSIS_DIR/commits/repo_commits.json`. Then, the API analysis results are saved in `ANALYSIS_DIR/apis/repo_ac_map.json`. You can use the `--no-grep` flag to disable the grep-based filtering, the `--max_year` flag to filter commits by year, and `--max-commits N` to limit the newest matching candidate commits analyzed. The CLI value overrides the optional top-level YAML setting `max_commits`.


### 3. Generate performance tests

Run the following to generate performance tests for the configured experiment:
```bash
python src/gso/collect/generate/generate.py /path/to/experiment.yaml
```

Remember to set [`GHAPI_TOKEN`](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) env var. Creates an experiment workspace in `EXPERIMENTS_DIR/{exp_id}` and moves your configuration file there. It then generates performance tests for the configured experiment and saves it in the workspace as `{exp_id}_problems.json`.

### 4. Execute performance tests

*Prerequisite*: Cloud credentials set up for `skypilot` to spin up machines.
Run `sky check` and follow the instructions it provides to set up credentials.
Then, run the following to execute the generated performance tests:
```bash
python src/gso/collect/execute/execute.py --exp_id repo --machines K
```

This runs performance tests for the configured experiment on `K` machines and saves results in the workspace in `{exp_id}_results.json`. Optionally use `--api` to run tests for a single API. Use `--interactive` to run tests in interactive mode (for debugging).

View the stats of the results using:
```bash
python src/gso/collect/execute/evaluate.py --exp_id repo
```

For local validation, use an image named `gso-{exp_id}`. The image must contain
the full repository checkout at `/workspace/{repo_name}` (including `.git` and
candidate commits). `commits.py` clones the repository, when it is not already
present, to `~/buckets/gso_bucket/analysis/repos/{repo_name}`. Reuse that
checkout when building the execution image:
```bash
# Build the base image once, if it has not already been built.
docker build \
  --platform linux/amd64 \
  -t gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  -f ../../dockerfiles/Dockerfile.gso-base-ubuntu22.04-py312-uv0.5.4-amd64 \
  ../..

python src/gso/collect/execute/execute.py \
  --backend docker \
  --exp_id numpy \
  --api numpy.add \
  --docker-base-image gso-base:ubuntu22.04-py312-uv0.5.4-amd64 \
  --docker-repo-path ~/buckets/gso_bucket/analysis/repos/numpy \
  --docker-image gso-numpy:latest \
  --docker-platform linux/amd64 \
  --machines 1
```

`--docker-repo-path` requires `--docker-base-image`, and the path must contain a
`.git` directory. GSO copies the checkout, including local working-tree files,
into `/workspace/{repo_name}` in the image. Pass `--rebuild-docker-image` to
bypass Docker's build cache. This rebuilds from the current local checkout; it
does not fetch or pull the repository. Update the checkout explicitly when
new remote commits are needed:
```bash
git -C ~/buckets/gso_bucket/analysis/repos/numpy fetch --all --tags --prune
```

Omit `--docker-repo-path` to retain the remote-clone build behavior. Without
`--docker-base-image`, `--docker-image` must already contain the repository
checkout.

Docker results and container logs are saved separately as
`{exp_id}_results_docker.json` and `docker_logs/` in the experiment directory.
Analyze the local parent-to-candidate comparison with:
```bash
python src/gso/collect/execute/evaluate.py \
  --backend docker \
  --exp_id numpy \
  --api numpy.add
```

### 5. Bulk pipeline across many repositories

`run_pipeline.py` orchestrates the full collection workflow (commit analysis →
API mapping → test generation → execution → evaluation/dataset build) for every
repository in a CSV list, running several repositories concurrently. Each
repository gets its own workspace under `experiments/{repo}/` (an auto-generated
`experiment.yaml`, run logs under `logs/`, and evaluation plots under `plots/`),
while GSO-generated artifacts remain in the configurable GSO bucket
(`~/buckets/gso_bucket` by default).

For each repository, the evaluation stage also selects `(pid, commit)` pairs
whose measured speedup meets the dataset threshold, writes
`experiments/{exp_id}/{exp_id}_pids.py` inside the bucket, and builds
`datasets/gso_{exp_id}_dataset.jsonl`. If no pair meets the threshold, it logs
that the PID export and dataset build were skipped.

```bash
.venv/bin/python src/gso/collect/run_pipeline.py \
  assets/gso-python-performance-repositories \
  -j 3 --max-year 2022 -n 5
```

The config is rendered from the [experiment template](/assets/experiment.yaml)
with `exp_id`/`repo_url` filled in; existing configs are reused unless
`--overwrite-config` is passed. Stage selection (`--stages`) lets you run pieces
independently — for example analysis only, or generation on its own (which
transparently reuses the `analysis/commits` and `analysis/apis` artifacts
already produced in the bucket):
```bash
.venv/bin/python src/gso/collect/run_pipeline.py assets/gso-python-performance-repositories \
  --only numpy,markitdown --stages commits,apis
.venv/bin/python src/gso/collect/run_pipeline.py assets/gso-python-performance-repositories \
  --only numpy,markitdown --stages generate,execute,evaluate
```

Relocate the GSO bucket with `--buckets-dir` (or `GSO_BUCKET_DIR`), restrict to
a few repositories with `--only`, restrict to one API with `--api`, and preview
the exact commands with `--dry-run`. See `--help` for the full option list.
Run with the GSO virtualenv interpreter (`.venv/bin/python`) so the spawned
GSO subprocesses can `import gso`.

To distribute repositories across multiple LLM credentials, store each key in
its own environment variable and pass the variable names with
`--api-key-envs`. The runner assigns them round-robin in the filtered CSV order
and writes only the environment variable name to each YAML (never the key):

```bash
.venv/bin/python src/gso/collect/run_pipeline.py assets/gso-python-performance-repositories \
  --api-key-envs LLM_API_KEY_1,LLM_API_KEY_2,LLM_API_KEY_3
```

For example, repositories 1 and 4 use `LLM_API_KEY_1`. The option is also
repeatable (`--api-key-env LLM_API_KEY_1 --api-key-env LLM_API_KEY_2`). When a
config already exists, only its `llm.api_key_env` entry is updated; use
`--overwrite-config` separately when the entire template should be rendered
again.

When a stage is selected on its own (its producer stage is not in `--stages`),
the runner reuses the producer's artifact from the bucket. If that artifact is
missing — for example `generate` without a prior `analysis/apis/{repo}_ac_map.json`
or `execute` without `experiments/{exp_id}/{exp_id}_problems.json` — the stage is
skipped with a clear warning and a hint (`run --stages ... first`) instead of
letting `generate.py`/`execute.py` crash with a `FileNotFoundError`. Downstream
stages are skipped transitively, so a skipped `generate` also skips `execute`
and `evaluate`. A skipped repo counts as `SKIP(...)` in the summary (exit code 0),
not a failure; a repo whose selected stage actually ran and exited non-zero is
reported as `FAIL@<stage>` (exit code 1).
