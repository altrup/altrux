import json
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "protect_paths.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="the rented box has no git")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "PROTECTED_PATHS").write_text("a.py\n")
    (tmp_path / "a.py").write_text("1\n")
    (tmp_path / "b.py").write_text("1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "a.py").write_text("2\n")
    (tmp_path / "b.py").write_text("2\n")
    return tmp_path


def _hook(repo: Path, tool: str, tool_input: dict[str, str]) -> int:
    payload = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(repo)})
    return subprocess.run(
        ["python3", str(HOOK)],
        input=payload,
        text=True,
        cwd=repo,
        env={"CLAUDE_PROJECT_DIR": str(repo), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=False,
    ).returncode


def test_commit_with_protected_file_staged_is_refused(repo: Path) -> None:
    _git(repo, "add", "a.py")
    assert _hook(repo, "Bash", {"command": "git commit -m x"}) == 2


def test_commit_with_only_other_files_staged_is_allowed(repo: Path) -> None:
    _git(repo, "add", "b.py")
    assert _hook(repo, "Bash", {"command": "git commit -m x"}) == 0


def test_commit_all_with_protected_file_modified_is_refused(repo: Path) -> None:
    assert _hook(repo, "Bash", {"command": "git commit -am x"}) == 2


def test_commit_pathspec_naming_protected_file_is_refused(repo: Path) -> None:
    assert _hook(repo, "Bash", {"command": "git commit -m x -- a.py"}) == 2


def test_commit_pathspec_naming_other_file_is_allowed(repo: Path) -> None:
    assert _hook(repo, "Bash", {"command": "git commit -m x b.py"}) == 0


def test_non_commit_bash_is_allowed(repo: Path) -> None:
    _git(repo, "add", "a.py")
    assert _hook(repo, "Bash", {"command": "git status"}) == 0


def test_edit_of_protected_file_is_allowed(repo: Path) -> None:
    assert _hook(repo, "Edit", {"file_path": str(repo / "a.py")}) == 0


def test_edit_of_the_hook_itself_is_refused(repo: Path) -> None:
    assert _hook(repo, "Edit", {"file_path": str(repo / "PROTECTED_PATHS")}) == 2
