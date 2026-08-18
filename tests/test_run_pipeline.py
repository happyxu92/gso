import pytest

from gso.collect.run_pipeline import (
    assign_api_key_envs,
    normalize_api_key_envs,
    prepare_workspace,
    render_config,
)


TEMPLATE = """\
exp_id: "REPLACE_EXP_ID"
repo_url: "https://github.com/REPLACE_OWNER/REPLACE_REPOSITORY"
llm:
  model_name: "test-model"
  api_key_env: "LLM_API_KEY"  # credential source
"""


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
