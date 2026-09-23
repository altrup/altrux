#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/pristine/sft/training" "$tmp/altrux/sft/training" "$tmp/altrux/sft/.venv" \
  "$tmp/altrux/sft/data" "$tmp/altrux/sft/logs" "$tmp/altrux/sft/__pycache__" "$tmp/altrux/scripts"
echo 'x = 1' > "$tmp/pristine/sft/training/loop.py"
echo 'x = 2' > "$tmp/altrux/sft/training/loop.py"
echo 'new' > "$tmp/altrux/sft/training/added.py"
echo junk > "$tmp/altrux/sft/.venv/lib.py"; echo junk > "$tmp/altrux/sft/data/big.pt"
echo junk > "$tmp/altrux/sft/logs/run.log"; echo junk > "$tmp/altrux/sft/__pycache__/a.pyc"
cp "$ROOT/scripts/box_patch.sh" "$tmp/altrux/scripts/"

out="$(cd "$tmp/altrux" && PRISTINE="$tmp/pristine" bash scripts/box_patch.sh fix-loop)"
patch="$tmp/altrux/notes/experiments/patches/$(ls "$tmp/altrux/notes/experiments/patches")"
[[ "$out" == *"$patch"* ]]
grep -q '^+x = 2' "$patch"
grep -q 'added.py' "$patch"
! grep -qE '\.venv|/data/|/logs/|__pycache__' "$patch"
# applies cleanly on top of the pristine tree
(cd "$tmp/pristine" && patch -p1 --dry-run < "$patch" >/dev/null)
[[ "$(basename "$patch")" == *-fix-loop.patch ]]

echo "box_patch tests passed"
