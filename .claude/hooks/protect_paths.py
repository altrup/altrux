#!/usr/bin/env python3
"""PreToolUse hook: deny Edit/Write on the hand-written files in PROTECTED_PATHS."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELF_PROTECTED = ("PROTECTED_PATHS", ".claude/settings.json", ".claude/hooks/protect_paths.py")


def main() -> int:
    if os.environ.get("ALTRUX_PROTECT_OFF"):
        return 0
    payload = json.load(sys.stdin)
    if payload.get("tool_name") not in ("Edit", "Write", "MultiEdit"):
        return 0
    raw = payload.get("tool_input", {}).get("file_path")
    if not raw:
        return 0
    target = Path(payload.get("cwd", ROOT), raw).resolve()
    listed = [l.strip() for l in (ROOT / "PROTECTED_PATHS").read_text().splitlines()]
    protected = {ROOT / p for p in listed + list(SELF_PROTECTED) if p and not p.startswith("#")}
    if target in protected:
        rel = target.relative_to(ROOT)
        print(f"{rel} is in PROTECTED_PATHS: hand-written code the person owns. "
              "Do not edit it; propose the change as a diff in the conversation.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
