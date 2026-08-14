---
name: build-gso-docker-task
description: Build and validate a GSO performance-optimization task for a Python repository with the local Docker backend, from experiment YAML and performance-commit/API analysis through Docker-validated test generation, formal parent-vs-candidate execution, evaluation, PID selection, and dataset JSONL creation. Use when you are asked to create, continue, debug, inspect, or document a GSO task pipeline; run commits.py, apis.py, generate.py with inline Docker validation, execute.py/evaluate.py with --backend docker; prepare custom_pids.py; or build a repository-specific GSO dataset from a local gso-base image.
---

# Build a GSO Docker Task

Build one repository at a time and treat every stage as a checkpoint. Run commands from the GSO repository root. Default to the local checkout at `third/gso` only when it exists; otherwise discover the root from the user's context or ask for it. Unless the user explicitly chooses another cutoff, analyze and generate tasks for every matching commit after 2022 by passing `--max_year 2022`; do not cap the commit count with `--max-commits`, `--max_commits`, or YAML `max_commits`.

For every new repository task, first create a repository-scoped operator workspace at `experiments/${repo_name}/`. Put the experiment YAML, captured command logs, evaluation plots, `custom_pids.py`, notes, and other task-maintained files there. This workspace is distinct from GSO's generated artifact directory at `~/buckets/gso_bucket/experiments/${exp_id}/`, which remains fixed by `src/gso/constants.py`.

Read [references/workflow.md](references/workflow.md) before running or debugging the pipeline. Read [references/artifacts.md](references/artifacts.md) when inspecting JSON outputs, selecting tasks, or diagnosing a missing/stale artifact.

## Guardrails

- Preserve the user's existing GSO worktree changes. Inspect `git status --short` before edits and do not reset, clean, or overwrite unrelated files.
- Do not scatter new task files in the repository root. Keep user-authored configuration and outputs under `experiments/${repo_name}/`; do not relocate GSO-generated bucket artifacts or an existing task's files without the user's approval.
- Never place API keys in YAML, logs, commands, or chat. Keep `llm.api_key_env` in the config and use the process environment or a `.env` file.
- Treat commit analysis, LLM generation, Docker image builds, generation-time test execution, and formal performance execution as potentially long-running or costly. For a request to create/configure the skill or explain commands, do not start those stages. For an explicit request to build a task, proceed stage by stage and report material failures.
- Verify prerequisites and inputs with read-only checks before each expensive stage. Do not infer success from a zero exit code alone; verify the expected artifact exists and contains the requested repository/API.
- Use the local analysis checkout with `--docker-repo-path` by default so the image contains the exact commits already analyzed. Fetch only when requested or when a required commit is demonstrably absent.
- Do not hand-edit generated results to make a task pass. Correct configuration, regenerate/re-execute, or exclude an unsuitable candidate.

## Resolve the experiment identity

Derive these values once and reuse them consistently:

- `repo_root`: GSO repository root containing `src/gso`.
- `repo_name`: basename of YAML `repo_url` with an optional `.git` suffix removed; this names the operator workspace, analysis artifacts, and local checkout. Do not assume `exp_id == repo_name`.
- `workspace_dir`: `${repo_root}/experiments/${repo_name}`. For a new task, use `${workspace_dir}/experiment.yaml`, `${workspace_dir}/logs/`, `${workspace_dir}/plots/`, and `${workspace_dir}/custom_pids.py`.
- `yaml_path`: experiment YAML path; for a new repository task this must be `${workspace_dir}/experiment.yaml`.
- `exp_id`: YAML `exp_id`; this names GSO-generated experiment directories and result files.
- `bucket`: `~/buckets/gso_bucket` as defined by `src/gso/constants.py`.
- `base_image`: default `gso-base:ubuntu22.04-py312-uv0.5.4-amd64` unless the user specifies another image.
- `repo_image`: default `gso-${exp_id}:latest`.
- `max_year`: default `2022`. Pass the same value to commit analysis and generation so every analyzed commit remains eligible for task generation.

Use the bundled helper for a deterministic dry-run summary:

```bash
python /path/to/build-gso-docker-task/scripts/gso_paths.py \
  /path/to/experiment.yaml \
  --repo-root /path/to/gso
```

Pass `--api package.target_api`, `--base-image ...`, or `--rebuild-docker-image` to render the matching narrow commands. This helper reads configuration only; it does not run GSO or Docker.

## Execute the workflow

1. Derive `repo_name` from the repository URL, create `${repo_root}/experiments/${repo_name}/{logs,plots}`, and use this as the task workspace before creating configuration or running analysis. Capture every pipeline command's stdout/stderr under `${workspace_dir}/logs/` with stage-specific names; use shell `pipefail` when piping through `tee` so logging cannot hide a failed command.
2. If the user asks for a new config, copy [assets/experiment.yaml](assets/experiment.yaml) to `${workspace_dir}/experiment.yaml` and replace every placeholder. Keep public-API and deterministic CPU-only guidance unless repository-specific needs justify changes.
3. Run commit analysis with `--max_year 2022` (or the user's explicit cutoff), without any commit-count limit, and verify `${repo_name}_commits.json`. The step clones the repository only when the analysis checkout does not already exist. In `commits.py`, this cutoff selects commits strictly after the given year.
4. Run API mapping with `repo_name`, not `exp_id`, and verify `${repo_name}_ac_map.json`.
5. Before generation, inspect the repository's packaging and bootstrap files and decide whether the YAML `install_commands` will install the repository and required runtime dependencies. If repository-specific commands are needed, update the YAML first. If the YAML omits `install_commands` and the standard `Problem` commands are sufficient, leave the field absent. Also inspect the base image, the local checkout's `.git`, and candidate plus parent commit availability: generation now requires Docker and prepares the repository image before its first LLM request.
6. Generate only after those checks. Pass the same `--max_year 2022` cutoff used for analysis, the local analysis checkout with `--docker-repo-path`, plus the intended image, base image, and platform. By default generate across all mapped APIs and all eligible commits; use `--api` only when the user requests one API or when explicitly iterating cheaply. For each commit, generation creates one scenario/test response at a time, runs it against the parent and candidate in Docker, and supplies execution failures to up to three semantic retries. Only successfully executed scenarios inform the next request. Verify `${exp_id}_problems.json`, generation-validation logs, and any retry/failure diagnostic JSON. Failed generation does not replace an existing valid problems file.
7. Run formal execution with the Docker backend after generation; generation-time execution is a correctness gate and does not create `${exp_id}_results_docker.json` or replace repeated performance measurement. Use `--rebuild-docker-image` only when the image must reflect changed checkout contents or the user explicitly requests a no-cache build. Keep captured execution output in `${workspace_dir}/logs/`; GSO's detailed Docker logs remain in its fixed bucket artifact directory.
8. Evaluate Docker results in `commit` mode, write plots to `${workspace_dir}/plots/`, and inspect both the captured summary and plots. Repeat generation/execution for weak or invalid APIs rather than accepting noise.
9. Select final `(pid, 7-character candidate commit)` pairs from measured Docker results and create `${workspace_dir}/custom_pids.py` based on [assets/custom_pids.py](assets/custom_pids.py). Do not modify the repository's shared `src/gso/collect/pids.py`.
10. Build the dataset with `--backend docker` and `${workspace_dir}/custom_pids.py`, verify the JSONL, and summarize task count plus any selected items that were rejected by thresholds.

## Diagnose by stage

- Missing analysis JSON: confirm `repo_name` derivation and the preceding command's output.
- Authentication or LLM error: confirm the variable named by `llm.api_key_env` is set without printing its value; confirm `base_url`, model availability, timeout, and concurrency.
- No generated API: inspect the API-commit map and match `--api` exactly.
- Generation repository image failure: inspect `docker_logs/generation_validation/repository_image/build.log`; confirm the base image, platform, and self-contained local `.git` checkout. For a later formal-execution rebuild, inspect `docker_logs/repository_image/build.log`.
- Generated scenario/test rejection: inspect `${exp_id}_generation_retries_<timestamp>.json` for recovered attempts or `${exp_id}_generation_failed_<timestamp>.json` for terminal failures, then correlate `execution_log` with `docker_logs/generation_validation/`. Distinguish response-format/static-validation errors from setup, reference, candidate, or equivalence failures.
- Candidate validation failure: use `git -C <checkout> cat-file -e '<hash>^{commit}'` and `git -C <checkout> cat-file -e '<hash>^'`. Fetch deliberately if the analysis JSON refers to an object missing from the checkout.
- No usable execution results: inspect the task-specific directory under `docker_logs/`, then validate install commands, generated test determinism, phase scripts, and resource limits.
- Empty dataset: confirm the exact `(pid, commit)` selection exists in Docker results and satisfies the builder's speedup/test thresholds. Run `build_dataset.py --debug` to print missing selections.

## Completion criteria

Call the build complete only when all requested stages have verified artifacts. For a full task build, require:

- analysis commit JSON and API map;
- Docker-validated generated problems and generation-validation diagnostics when retries occurred;
- Docker results and logs;
- evaluation in parent-to-candidate (`commit`) mode;
- a syntactically valid custom PID file using 7-character hashes;
- a non-empty `~/buckets/gso_bucket/datasets/gso_${exp_id}_dataset.jsonl` unless the user explicitly accepts an empty selection.

Report the repository workspace path, exact generated artifact paths, commands that failed or were intentionally skipped, the selected PIDs/commits, and the final dataset row count. Never claim that LLM, Docker, or performance stages ran when only the helper rendered commands.
