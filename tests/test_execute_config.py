from pathlib import Path

import yaml

from gso.collect.execute.execute import _apply_experiment_overrides
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
