#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hook="$ROOT/.claude/hooks/protect_paths.py"
first="$(head -1 "$ROOT/PROTECTED_PATHS")"

run() { # tool file_path -> exit code
  printf '{"tool_name":"%s","cwd":"%s","tool_input":{"file_path":"%s"}}' "$1" "$ROOT" "$2" \
    | python3 "$hook" >/dev/null 2>&1 && echo 0 || echo $?
}

[[ "$(run Edit "$ROOT/$first")" == 2 ]]
[[ "$(run Write "$first")" == 2 ]]                       # relative to cwd
[[ "$(run Edit "$ROOT/PROTECTED_PATHS")" == 2 ]]         # the list itself
[[ "$(run Edit "$ROOT/.claude/hooks/protect_paths.py")" == 2 ]]
[[ "$(run Edit "$ROOT/.claude/settings.json")" == 2 ]]
[[ "$(run Edit "$ROOT/README.md")" == 0 ]]
[[ "$(run Edit "$ROOT/sft/training/cli.py")" == 0 ]]
[[ "$(run Bash "$ROOT/$first")" == 0 ]]                  # only Edit/Write are checked
[[ "$(ALTRUX_PROTECT_OFF=1 run Edit "$ROOT/$first")" == 0 ]]
printf '{"tool_name":"Edit","tool_input":{}}' | python3 "$hook"   # no file_path -> allow

# the deny reason names the file
printf '{"tool_name":"Edit","cwd":"%s","tool_input":{"file_path":"%s"}}' "$ROOT" "$first" \
  | { python3 "$hook" 2>&1 >/dev/null || true; } | grep -q "$first"

echo "protect_paths tests passed"
