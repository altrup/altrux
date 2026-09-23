#!/usr/bin/env bash
# On the rented box: write the working tree's changes against the uploaded
# pristine copy as one patch under notes/experiments/patches/, where the
# watchdog pull carries it home for review. There is no git on the box.
#
#   scripts/box_patch.sh <name>      -> notes/experiments/patches/<UTC>-<name>.patch
set -euo pipefail
name="${1:?usage: box_patch.sh <name>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRISTINE="${PRISTINE:-$HOME/pristine}"
out="$REPO/notes/experiments/patches/$(date -u +%Y%m%d-%H%M%S)-$name.patch"
mkdir -p "$(dirname "$out")"
# diff exits 1 when files differ, which is the point.
(cd "$(dirname "$PRISTINE")" && diff -ruN \
  -x .venv -x .cache -x data -x logs -x checkpoints -x notes -x __pycache__ -x '*.pyc' -x '*.pt' \
  "$(basename "$PRISTINE")" "$(realpath --relative-to="$(dirname "$PRISTINE")" "$REPO")" ) > "$out" || [[ $? == 1 ]]
[[ -s "$out" ]] || { rm -f "$out"; echo "no changes against $PRISTINE" >&2; exit 1; }
echo "$out"
