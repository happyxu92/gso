---
name: build-gso-docker-task
description: Build and validate a GSO performance-optimization task for a Python repository with the local Docker backend, from experiment YAML and performance-commit/API analysis through generated tests, parent-vs-candidate execution, evaluation, PID selection, and dataset JSONL creation. Use when you are asked to create, continue, debug, inspect, or document a GSO task pipeline; run commits.py, apis.py, generate.py, execute.py/evaluate.py with --backend docker; prepare custom_pids.py; or build a repository-specific GSO dataset from a local gso-base image.
---

# Build a GSO Docker Task

Build one repository at a time and treat every stage as a checkpoint. Run commands from the GSO repository root. Default to the local checkout at `third/gso` only when it exists; otherwise discover the root from the user's context or ask for it.

Read [references/workflow.md](references/workflow.md) before running or debugging the pipeline. Read [references/artifacts.md](references/artifacts.md) when inspecting JSON outputs, selecting tasks, or diagnosing a missing/stale artifact.

## Guardrails

- Preserve the user's existing GSO worktree changes. Inspect `git status --short` before edits and do not reset, clean, or overwrite unrelated files.
- Never place API keys in YAML, logs, commands, or chat. Keep `llm.api_key_env` in the config and use the process environment or a `.env` file.
- Treat commit analysis, LLM generation, Docker image builds, and performance execution as potentially long-running or costly. For a request to create/configure the skill or explain commands, do not start those stages. For an explicit request to build a task, proceed stage by stage and report material failures.
- Verify prerequisites and inputs with read-only checks before each expensive stage. Do not infer success from a zero exit code alone; verify the expected artifact exists and contains the requested repository/API.
- Use the local analysis checkout with `--docker-repo-path` by default so the image contains the exact commits already analyzed. Fetch only when requested or when a required commit is demonstrably absent.
- Do not hand-edit generated results to make a task pass. Correct configuration, regenerate/re-execute, or exclude an unsuitable candidate.

## Resolve the experiment identity

Derive these values once and reuse them consistently:

- `repo_root`: GSO repository root containing `src/gso`.
- `yaml_path`: experiment YAML path.
- `exp_id`: YAML `exp_id`; this names the experiment directory and result files.
- `repo_name`: basename of YAML `repo_url` with an optional `.git` suffix removed; this names analysis artifacts and the local checkout. Do not assume `exp_id == repo_name`.
- `bucket`: `~/buckets/gso_bucket` as defined by `src/gso/constants.py`.
- `base_image`: default `gso-base:ubuntu22.04-py312-uv0.5.4-amd64` unless the user specifies another image.
- `repo_image`: default `gso-${exp_id}:latest`.

Use the bundled helper for a deterministic dry-run summary:

```bash
python /path/to/build-gso-docker-task/scripts/gso_paths.py \
  /path/to/experiment.yaml \
  --repo-root /path/to/gso
```

Pass `--api package.target_api`, `--base-image ...`, or `--rebuild-docker-image` to render the matching narrow commands. This helper reads configuration only; it does not run GSO or Docker.

## Execute the workflow

2. If the user asks for a new config, copy [assets/experiment.yaml](assets/experiment.yaml) to the requested location and replace every placeholder. Keep public-API and deterministic CPU-only guidance unless repository-specific needs justify changes.
3. Run commit analysis and verify `${repo_name}_commits.json`. The step clones the repository only when the analysis checkout does not already exist.
4. Run API mapping with `repo_name`, not `exp_id`, and verify `${repo_name}_ac_map.json`.
5. Before generation, inspect the repository's packaging and bootstrap files and decide whether the YAML `install_commands` will install the repository and required runtime dependencies. If repository-specific commands are needed, update the YAML first. If the YAML omits `install_commands` and the standard `Problem` commands are sufficient, leave the field absent. Generate only after this check; generation prefers YAML commands and otherwise embeds the defaults. Use `--api` when the user requests one API or when iterating cheaply. Verify `${exp_id}_problems.json`; failed generation diagnostics do not replace an existing valid problems file.
6. Inspect the base image, the local checkout's `.git`, and candidate commit availability. Execute with the Docker backend. Use `--rebuild-docker-image` only when the image must reflect changed checkout contents or the user explicitly requests a no-cache build.
7. Evaluate Docker results in `commit` mode and inspect both the printed summary and plots. Repeat generation/execution for weak or invalid APIs rather than accepting noise.
8. Select final `(pid, 7-character candidate commit)` pairs from measured Docker results and create a separate `custom_pids.py` based on [assets/custom_pids.py](assets/custom_pids.py). Do not modify the repository's shared `src/gso/collect/pids.py`.
9. Build the dataset with `--backend docker`, verify the JSONL, and summarize task count plus any selected items that were rejected by thresholds.

## Diagnose by stage

- Missing analysis JSON: confirm `repo_name` derivation and the preceding command's output.
- Authentication or LLM error: confirm the variable named by `llm.api_key_env` is set without printing its value; confirm `base_url`, model availability, timeout, and concurrency.
- No generated API: inspect the API-commit map and match `--api` exactly.
- Docker repository image failure: inspect `docker_logs/repository_image/build.log`; confirm the base image, platform, and self-contained local `.git` checkout.
- Candidate validation failure: use `git -C <checkout> cat-file -e '<hash>^{commit}'` and `git -C <checkout> cat-file -e '<hash>^'`. Fetch deliberately if the analysis JSON refers to an object missing from the checkout.
- No usable execution results: inspect the task-specific directory under `docker_logs/`, then validate install commands, generated test determinism, phase scripts, and resource limits.
- Empty dataset: confirm the exact `(pid, commit)` selection exists in Docker results and satisfies the builder's speedup/test thresholds. Run `build_dataset.py --debug` to print missing selections.

## Completion criteria

Call the build complete only when all requested stages have verified artifacts. For a full task build, require:

- analysis commit JSON and API map;
- validated generated problems;
- Docker results and logs;
- evaluation in parent-to-candidate (`commit`) mode;
- a syntactically valid custom PID file using 7-character hashes;
- a non-empty `~/buckets/gso_bucket/datasets/gso_${exp_id}_dataset.jsonl` unless the user explicitly accepts an empty selection.

Report exact artifact paths, commands that failed or were intentionally skipped, the selected PIDs/commits, and the final dataset row count. Never claim that LLM, Docker, or performance stages ran when only the helper rendered commands.
