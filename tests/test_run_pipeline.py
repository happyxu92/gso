import re
import sys
from pathlib import Path

import pytest

from gso.collect.run_pipeline import (
    assign_api_key_envs,
    build_stages,
    compute_paths,
    normalize_api_key_envs,
    parse_args,
    prepare_workspace,
    render_config,
    run_command,
    timestamped,
)


TEMPLATE = """\
exp_id: "REPLACE_EXP_ID"
repo_url: "https://github.com/REPLACE_OWNER/REPLACE_REPOSITORY"
llm:
  model_name: "test-model"
  api_key_env: "LLM_API_KEY"  # credential source
"""
TIMESTAMP_RE = r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00\]"


def test_parse_args_uses_asset_template_and_300_commit_default():
    args = parse_args(["repositories.csv"])

    assert (
        args.template == Path(__file__).resolve().parents[1] / "assets/experiment.yaml"
    )
    assert args.max_commits == 300
    assert args.test_timeout == 300


def test_pipeline_passes_test_timeout_to_execute_stage(tmp_path):
    args = parse_args(["repositories.csv", "--test-timeout", "120"])
    paths = compute_paths(tmp_path / "buckets")

    stages = build_stages(
        repo="demo",
        exp_id="demo",
        repo_checkout=tmp_path / "checkout",
        config_path=tmp_path / "demo.yaml",
        plots_dir=tmp_path / "plots",
        args=args,
        paths=paths,
    )

    execute_command = next(
        command for name, command, _ in stages if name == "execute"
    )
    timeout_index = execute_command.index("--test-timeout")
    assert execute_command[timeout_index + 1] == "120"


def test_pipeline_rejects_non_positive_test_timeout():
    with pytest.raises(SystemExit):
        parse_args(["repositories.csv", "--test-timeout", "0"])


def test_pipeline_evaluate_stage_builds_dataset(tmp_path):
    args = parse_args(["repositories.csv"])
    paths = compute_paths(tmp_path / "buckets")

    stages = build_stages(
        repo="demo",
        exp_id="demo",
        repo_checkout=tmp_path / "checkout",
        config_path=tmp_path / "demo.yaml",
        plots_dir=tmp_path / "plots",
        args=args,
        paths=paths,
    )

    evaluate_command = next(
        command for name, command, _ in stages if name == "evaluate"
    )
    assert "--build-dataset" in evaluate_command


def test_pipeline_lowercases_only_the_docker_image_name(tmp_path):
    args = parse_args(["repositories.csv"])
    paths = compute_paths(tmp_path / "buckets")

    stages = build_stages(
        repo="MinerU",
        exp_id="MinerU",
        repo_checkout=tmp_path / "MinerU",
        config_path=tmp_path / "MinerU.yaml",
        plots_dir=tmp_path / "MinerU-plots",
        args=args,
        paths=paths,
    )

    image_names = []
    for _, command, _ in stages:
        if "--docker-image" in command:
            image_names.append(command[command.index("--docker-image") + 1])

    assert image_names
    assert set(image_names) == {"gso-mineru:latest"}
    execute_command = next(
        command for name, command, _ in stages if name == "execute"
    )
    assert execute_command[execute_command.index("--exp_id") + 1] == "MinerU"


def test_timestamped_uses_beijing_iso_8601_time():
    assert re.fullmatch(rf"{TIMESTAMP_RE} message", timestamped("message"))


def test_normalize_api_key_envs_accepts_commas_and_repeated_values():
    assert normalize_api_key_envs(["KEY_1, KEY_2", "KEY_3"]) == [
        "KEY_1",
        "KEY_2",
        "KEY_3",
    ]


@pytest.mark.parametrize("values", [["KEY-1"], ["KEY_1,,KEY_2"], ["KEY_1", "KEY_1"]])
def test_normalize_api_key_envs_rejects_invalid_values(values):
    with pytest.raises(SystemExit):
        normalize_api_key_envs(values)


def test_assign_api_key_envs_round_robins_in_input_order():
    rows = [
        ("one", "https://example.test/one"),
        ("two", "https://example.test/two"),
        ("three", "https://example.test/three"),
    ]

    assert assign_api_key_envs(rows, ["KEY_1", "KEY_2"]) == [
        (rows[0], "KEY_1"),
        (rows[1], "KEY_2"),
        (rows[2], "KEY_1"),
    ]


def test_render_config_sets_assigned_api_key_env():
    rendered = render_config(
        TEMPLATE,
        "demo",
        "https://github.com/example/demo",
        "LLM_API_KEY_2",
    )

    assert 'exp_id: "demo"' in rendered
    assert 'repo_url: "https://github.com/example/demo"' in rendered
    assert 'api_key_env: "LLM_API_KEY_2"  # credential source' in rendered
    assert "LLM_API_KEY_1" not in rendered


def test_prepare_workspace_only_updates_key_in_existing_config(tmp_path):
    workspace = tmp_path / "experiments" / "demo"
    logs_dir = workspace / "logs"
    plots_dir = workspace / "plots"
    workspace.mkdir(parents=True)
    config_path = workspace / "demo.yaml"
    customized = TEMPLATE.replace('model_name: "test-model"', 'model_name: "custom"')
    config_path.write_text(customized, encoding="utf-8")

    result = prepare_workspace(
        repo="demo",
        url="https://github.com/example/demo",
        repository="example/demo",
        workspace=workspace,
        logs_dir=logs_dir,
        plots_dir=plots_dir,
        template_text=TEMPLATE,
        overwrite_config=False,
        dry_run=False,
        api_key_env="LLM_API_KEY_2",
    )

    updated = result.read_text(encoding="utf-8")
    assert 'model_name: "custom"' in updated
    assert 'api_key_env: "LLM_API_KEY_2"  # credential source' in updated


def test_run_command_relays_generation_output_without_verbose(tmp_path, capsys):
    log_path = tmp_path / "generate.log"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; print('ordinary subprocess output'); "
            "sys.stderr.write('\\rGenerating commit tests: 0%'); "
            "sys.stderr.flush(); "
            "print('Generation event: repo.api/abcdef1 scenario/test 1 requesting test', "
            "file=sys.stderr, flush=True); "
            "print('Generation progress: repo.api/abcdef1 test 1/2 accepted', "
            "file=sys.stderr, flush=True)"
        ),
    ]

    return_code, _ = run_command(command, log_path, "repo", verbose=False)

    assert return_code == 0
    console_lines = capsys.readouterr().out.splitlines()
    assert len(console_lines) == 2
    assert re.fullmatch(
        rf"{TIMESTAMP_RE} \[repo\] Generation event: "
        r"repo\.api/abcdef1 scenario/test 1 requesting test",
        console_lines[0],
    )
    assert re.fullmatch(
        rf"{TIMESTAMP_RE} \[repo\] Generation progress: "
        r"repo\.api/abcdef1 test 1/2 accepted",
        console_lines[1],
    )
    console = "\n".join(console_lines)
    assert "ordinary subprocess output" not in console
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines
    assert all(re.match(rf"{TIMESTAMP_RE} ", line) for line in log_lines)
    log_text = "\n".join(log_lines)
    assert "ordinary subprocess output" in log_text
    assert (
        "Generation event: repo.api/abcdef1 scenario/test 1 requesting test" in log_text
    )
    assert "Generation progress: repo.api/abcdef1 test 1/2 accepted" in log_text


def test_run_command_timestamps_verbose_terminal_output(tmp_path, capsys):
    log_path = tmp_path / "commits.log"

    return_code, _ = run_command(
        [sys.executable, "-c", "print('stage output')"],
        log_path,
        "repo",
        verbose=True,
    )

    assert return_code == 0
    assert re.fullmatch(
        rf"{TIMESTAMP_RE} \[repo\] stage output\n",
        capsys.readouterr().out,
    )
    assert all(
        re.match(rf"{TIMESTAMP_RE} ", line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    )
