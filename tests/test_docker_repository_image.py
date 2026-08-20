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
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    assert (
        'git clone --recursive "$REPO_URL" "/workspace/$REPO_NAME"' in dockerfile_text
    )
    assert "apt-get install -y --no-install-recommends ccache" in dockerfile_text
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


def test_launch_task_mounts_persistent_build_and_package_cache(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_name = "repo-api_task.yaml"
    (workspace / task_name).write_text("run: echo ready\n", encoding="utf-8")
    cache_dir = tmp_path / "repo-cache"
    manager = DockerManager(
        image="gso-repo:latest",
        artifact_dir=tmp_path / "artifacts",
        cache_dir=cache_dir,
    )
    commands = []

    def fake_run(command, *, check=True, capture_output=False):
        commands.append(command)
        if command[:3] == ["docker", "container", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(manager, "_run", fake_run)

    manager.launch_task(task_name, workspace, cluster="docker-gso-cache-test")

    create_command = next(
        command for command in commands if command[:2] == ["docker", "create"]
    )
    mount_index = create_command.index("--mount")
    assert create_command[mount_index + 1] == (
        f"type=bind,source={cache_dir.resolve()},target=/gso-cache"
    )
    assert "CCACHE_DIR=/gso-cache/ccache" in create_command
    assert "PIP_CACHE_DIR=/gso-cache/pip" in create_command
    assert "UV_CACHE_DIR=/gso-cache/uv" in create_command
    assert not any("XDG_CACHE" in argument for argument in create_command)
    assert not (cache_dir / "xdg").exists()
    shell_command = create_command[-1]
    assert "CMAKE_CXX_COMPILER_LAUNCHER=ccache" in shell_command
    assert "USE_CCACHE=1" in shell_command
    assert (cache_dir / "ccache").is_dir()


def test_prepare_commit_image_installs_parent_once_without_xdg_cache(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    manager = DockerManager(
        image="gso-repo:latest",
        artifact_dir=tmp_path / "artifacts",
        cache_dir=cache_dir,
    )
    problem = SimpleNamespace(
        pid="repo-api",
        repo=SimpleNamespace(repo_name="repo"),
        install_commands=[
            "uv venv --python 3.12",
            "source .venv/bin/activate",
            "uv pip install -e .",
        ],
    )
    commands = []

    def fake_run(command, *, check=True, capture_output=False):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(manager, "_run", fake_run)

    image = manager.prepare_commit_image(
        problem,
        "1234567890abcdef",
        session_id="session-1",
    )

    assert image.startswith("gso-repo:prepared-session-1-1234567890ab-")
    create_command = next(
        command for command in commands if command[:2] == ["docker", "create"]
    )
    assert not any("XDG_CACHE" in argument for argument in create_command)
    exec_command = next(
        command for command in commands if command[:2] == ["docker", "exec"]
    )
    install_script = exec_command[-1]
    assert "git checkout --detach '1234567890abcdef^'" in install_script
    assert "uv pip install -e ." in install_script
    assert "/gso-prepared-env.sh" in install_script
    assert any(command[:2] == ["docker", "commit"] for command in commands)
    assert commands[-1][:3] == ["docker", "rm", "--force"]


def test_phase1_result_collection_skips_missing_results_directory(
    tmp_path, monkeypatch
):
    artifact_dir = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    manager = DockerManager(image="gso-repo:latest", artifact_dir=artifact_dir)
    commands = []

    def fake_run(command, *, check=True, capture_output=False):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="phase 1 output\n", stderr="")

    monkeypatch.setattr(manager, "_run", fake_run)

    message, results = manager.get_results(
        workspace,
        "docker-gso-phase1",
        expect_results=False,
    )

    assert commands == [["docker", "logs", "docker-gso-phase1"]]
    assert results == []
    assert "phase-1 validation" in message
    assert "result collection skipped (not expected)" in message
    assert "result copy failed" not in message
    assert not (workspace / "results").exists()
    assert (artifact_dir / "docker-gso-phase1.log").read_text() == "phase 1 output\n"
