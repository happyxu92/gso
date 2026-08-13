from pathlib import Path
from types import SimpleNamespace

import pytest

from gso.collect.execute.dockermgr import DockerManager


def make_problem(repo_url="https://github.com/example/repo", repo_name="repo"):
    return SimpleNamespace(repo=SimpleNamespace(repo_url=repo_url, repo_name=repo_name))


def test_build_repository_image_clones_experiment_repository(tmp_path, monkeypatch):
    manager = DockerManager(
        image="gso-repo:latest",
        artifact_dir=tmp_path,
        platform="linux/amd64",
        repository_base_image="gso-base:latest",
    )
    commands = []

    def fake_run(command, *, check=True, capture_output=False):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="built\n", stderr="")

    monkeypatch.setattr(manager, "_run", fake_run)

    manager.build_repository_image([make_problem()])

    assert commands[0] == ["docker", "image", "inspect", "gso-base:latest"]
    build_command = commands[1]
    assert build_command[:4] == ["docker", "build", "--platform", "linux/amd64"]
    assert "BASE_IMAGE=gso-base:latest" in build_command
    assert "REPO_URL=https://github.com/example/repo" in build_command
    assert "REPO_NAME=repo" in build_command
    assert build_command[-1] == str(tmp_path / "repository_image")

    dockerfile = tmp_path / "repository_image" / "Dockerfile"
    assert 'git clone --recursive "$REPO_URL" "/workspace/$REPO_NAME"' in (
        dockerfile.read_text(encoding="utf-8")
    )
    assert (tmp_path / "repository_image" / "build.log").read_text(
        encoding="utf-8"
    ) == "built\n"


def test_build_repository_image_uses_local_repository(tmp_path, monkeypatch):
    local_repo = tmp_path / "source"
    (local_repo / ".git").mkdir(parents=True)
    (local_repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    manager = DockerManager(
        image="gso-repo:latest",
        artifact_dir=artifact_dir,
        repository_base_image="gso-base:latest",
        repository_path=local_repo,
    )
    commands = []

    def fake_run(command, *, check=True, capture_output=False):
        commands.append(command)
        if command[:2] == ["docker", "build"]:
            context_dir = Path(command[-1])
            assert context_dir != artifact_dir / "repository_image"
            assert (context_dir / "repository" / ".git").is_dir()
            assert (context_dir / "repository" / "module.py").is_file()
        return SimpleNamespace(returncode=0, stdout="built\n", stderr="")

    monkeypatch.setattr(manager, "_run", fake_run)

    manager.build_repository_image([make_problem()])

    build_command = commands[1]
    assert "REPO_NAME=repo" in build_command
    assert not any(arg.startswith("REPO_URL=") for arg in build_command)
    dockerfile = artifact_dir / "repository_image" / "Dockerfile"
    assert "COPY repository /workspace/${REPO_NAME}" in dockerfile.read_text(
        encoding="utf-8"
    )


def test_local_repository_requires_base_image(tmp_path):
    manager = DockerManager(
        image="gso-repo:latest",
        artifact_dir=tmp_path,
        repository_path=tmp_path,
    )

    with pytest.raises(ValueError, match="requires --docker-base-image"):
        manager.build_repository_image([make_problem()])


def test_build_repository_image_requires_one_repository(tmp_path):
    manager = DockerManager(
        image="gso-repo:latest",
        artifact_dir=Path(tmp_path),
        repository_base_image="gso-base:latest",
    )

    with pytest.raises(ValueError, match="exactly one repository"):
        manager.build_repository_image(
            [
                make_problem(),
                make_problem("https://github.com/example/other", "other"),
            ]
        )
