import shutil
import subprocess
import sys

from gso.collect.execute.skymgr import SkyManager
from gso.data import Problem, Repo, Tests as CommitTests
from gso.harness.environment.patches import (
    apply_patch_requests,
    ensure_patch_dependencies,
)


def test_cached_requests_response_supports_streaming(tmp_path):
    test_code = """
url = 'https://example.test/cached.bin'
cached_content = b'abcdef'
with open(url_to_filename(url), 'wb') as cache_file:
    cache_file.write(cached_content)

response = requests.get(url)
assert response._content_consumed is True
assert response.headers['X-From-Cache'] == 'true'
assert list(response.iter_content(chunk_size=2)) == [b'ab', b'cd', b'ef']
response.close()
"""
    patched_test = apply_patch_requests(test_code).replace(
        "CACHE_DIR = '/.url_cache'",
        f"CACHE_DIR = {str(tmp_path)!r}",
    )

    subprocess.run(
        [sys.executable, "-c", patched_test],
        check=True,
        capture_output=True,
        text=True,
    )


def test_patch_dependencies_are_added_once_to_custom_install_commands():
    custom_commands = ["uv venv --python 3.12", "source .venv/bin/activate"]

    commands = ensure_patch_dependencies("repo-api", custom_commands)
    commands = ensure_patch_dependencies("repo-api", commands)

    assert commands == [*custom_commands, "uv pip install requests"]


def test_workspace_installs_dependencies_for_injected_patches():
    problem = Problem(
        pid="repo-api",
        repo=Repo(
            repo_url="https://github.com/example/repo",
            repo_owner="example",
            repo_name="repo",
        ),
        api="api",
        py_version="3.12",
        install_commands=[
            "uv venv --python 3.12",
            "source .venv/bin/activate",
            "uv pip install -e .",
        ],
        tests=[CommitTests(commit_hash="abcdef123456", samples=["print('test')"])],
    )

    workspace = SkyManager.create_workspace(problem, phase1_only=True)
    try:
        phase1 = (workspace / "phase1.sh").read_text()
        test = (workspace / "abcdef1" / "test_0.py").read_text()

        assert "uv pip install requests" in phase1
        assert "import requests" in test
    finally:
        shutil.rmtree(workspace)


def test_workspace_renders_configured_test_timeout(tmp_path):
    problem = Problem(
        pid="repo-api",
        repo=Repo(
            repo_url="https://github.com/example/repo",
            repo_owner="example",
            repo_name="repo",
        ),
        api="api",
        py_version="3.12",
        tests=[CommitTests(commit_hash="abcdef123456", samples=["print('test')"])],
    )

    workspace = SkyManager.create_workspace(problem, test_timeout=120)
    try:
        assert "timeout 120s python" in (workspace / "phase1.sh").read_text()
        assert "timeout 120s python" in (workspace / "phase2.sh").read_text()
    finally:
        shutil.rmtree(workspace)
