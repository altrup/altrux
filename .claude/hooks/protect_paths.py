#!/usr/bin/env python3
"""PreToolUse hook: refuse any `git commit` that would carry a file in PROTECTED_PATHS.

Agents may edit the protected files; the person reviews and commits them. The
hook also refuses Edit/Write on itself, its wiring, and the list.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
SELF_PROTECTED = ("PROTECTED_PATHS", ".claude/settings.json", ".claude/hooks/protect_paths.py")
COMMIT = re.compile(r"\bgit\b(?:\s+-\S+)*\s+commit\b")


def protected() -> set[Path]:
    listed = (ROOT / "PROTECTED_PATHS").read_text().split()
    return {ROOT / p for p in listed + list(SELF_PROTECTED) if not p.startswith("#")}


def git_paths(cwd: Path, *args: str) -> set[Path]:
    out = subprocess.run(
        ["git", "diff", "--name-only", *args], cwd=cwd, capture_output=True, text=True, check=False
    ).stdout
    return {ROOT / line for line in out.splitlines()}


def commit_paths(command: str, cwd: Path) -> set[Path]:
    """Files a `git commit` command would commit: the index, plus -a or pathspecs."""
    words = shlex.split(command, posix=True)
    paths = git_paths(cwd, "--cached")
    short_flags = (w[1:] for w in words if w.startswith("-") and not w.startswith("--"))
    if "--all" in words or any("a" in flags for flags in short_flags):
        paths |= git_paths(cwd)
    for word in words[words.index("commit") + 1 :]:
        if not word.startswith("-") and (cwd / word).is_file():
            paths.add((cwd / word).resolve())
    return paths


def main() -> int:
    if os.environ.get("ALTRUX_PROTECT_OFF"):
        return 0
    payload = json.load(sys.stdin)
    tool, tool_input = payload.get("tool_name"), payload.get("tool_input", {})
    cwd = Path(payload.get("cwd", ROOT))
    if tool in ("Edit", "Write", "MultiEdit"):
        raw = tool_input.get("file_path")
        target = Path(cwd, raw).resolve() if raw else None
        if target in {ROOT / p for p in SELF_PROTECTED}:
            print(
                f"{target.relative_to(ROOT)} is the protection itself; the person edits it.",
                file=sys.stderr,
            )
            return 2
        return 0
    if tool != "Bash" or not COMMIT.search(tool_input.get("command", "")):
        return 0
    hit = sorted(
        p.relative_to(ROOT) for p in commit_paths(tool_input["command"], cwd) & protected()
    )
    if hit:
        print(
            "Refused: this commit carries PROTECTED_PATHS files the person commits: "
            + ", ".join(map(str, hit))
            + ". Leave the whole change unstaged and list them under PROTECTED EDIT "
            "in your final message.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
