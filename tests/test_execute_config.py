from pathlib import Path

import yaml

from gso.collect.execute.execute import (
    GeneratedTestExecutionConfig,
    _apply_experiment_overrides,
    _create_runtime,
)
from gso.data import Problem, Repo


def make_problem() -> Problem:
    return Problem(
        pid="repo-api",
        repo=Repo(
            repo_url="https://github.com/example/repo",
            repo_owner="example",
            repo_name="repo",
        ),
        api="api",
        py_version="3.12",
    )


def test_experiment_template_uses_default_install_commands():
    template_path = Path(__file__).resolve().parents[1] / "assets/experiment.yaml"
    config = yaml.safe_load(template_path.read_text())

    assert config["install_commands"] == []


def test_default_docker_image_lowercases_exp_id_without_changing_it(tmp_path):
    config = GeneratedTestExecutionConfig(exp_id="MinerU")

    runtime = _create_runtime(config, tmp_path / config.exp_id)

    assert runtime.image == "gso-mineru:latest"
    assert config.exp_id == "MinerU"


def test_default_docker_cache_is_scoped_to_experiment(tmp_path):
    config = GeneratedTestExecutionConfig(exp_id="repo")

    runtime = _create_runtime(config, tmp_path / config.exp_id)

    assert runtime.cache_dir == (tmp_path / "repo" / "docker_cache").resolve()


def test_explicit_docker_cache_directory_is_used(tmp_path):
    cache_dir = tmp_path / "shared" / "repo"
    config = GeneratedTestExecutionConfig(
        exp_id="repo", docker_cache_dir=str(cache_dir)
    )

    runtime = _create_runtime(config, tmp_path / config.exp_id)

    assert runtime.cache_dir == cache_dir.resolve()


def test_empty_install_commands_preserve_problem_defaults():
    problem = make_problem()
    default_commands = list(problem.install_commands)

    _apply_experiment_overrides(
        [problem], {"target_commit": "stable", "install_commands": []}
    )

    assert problem.target_commit == "stable"
    assert problem.install_commands == default_commands


def test_nonempty_install_commands_replace_problem_defaults():
    problem = make_problem()
    custom_commands = ["uv pip install -e '.[test]'"]

    _apply_experiment_overrides(
        [problem], {"install_commands": custom_commands}
    )

    assert problem.install_commands == custom_commands
    assert problem.install_commands is not custom_commands
