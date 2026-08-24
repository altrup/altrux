#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/scripts" "$tmp/.cache/lama_ckl" "$tmp/.cache/wheels"
cp "$ROOT/scripts/lambda_cache_artifacts.sh" "$tmp/scripts/"
touch "$tmp/.cache/lama_ckl/manifest.json"
touch "$tmp/.cache/wheels/mamba-linux_aarch64.whl"
touch "$tmp/.cache/wheels/mamba-linux_x86_64.whl"

selected="$({
  LAMBDA_CACHE_ARTIFACTS='lama_ckl wheels/*aarch64.whl'
  source "$tmp/scripts/lambda_cache_artifacts.sh"
  printf '%s\n' "${cache_local_paths[@]#*/./}"
})"
grep -Fxq '.cache/lama_ckl' <<< "$selected"
grep -Fxq '.cache/wheels/mamba-linux_aarch64.whl' <<< "$selected"
if grep -Fq 'x86_64' <<< "$selected"; then
  echo "cache selection included an unselected wheel" >&2
  exit 1
fi

for unsafe in . '../secret' '/tmp/secret' '*'; do
  if LAMBDA_CACHE_ARTIFACTS="$unsafe" bash -c \
    'source "$1"' _ "$tmp/scripts/lambda_cache_artifacts.sh" 2>/dev/null; then
    echo "unsafe cache artifact accepted: $unsafe" >&2
    exit 1
  fi
done

grep -Fq 'source "$SCRIPT_DIR/lambda_cache_artifacts.sh"' "$ROOT/scripts/lambda_launch.sh"
grep -Fq 'source "$SCRIPT_DIR/lambda_cache_artifacts.sh"' "$ROOT/scripts/lambda_pull.sh"
grep -Fq 'UV_FIND_LINKS="$REPO_DIR/.cache/wheels"' "$ROOT/scripts/lambda_setup.sh"
grep -Fq 'LAMBDA_CACHE_ARTIFACTS=' "$ROOT/scripts/.env.example"

echo "lambda cache artifact tests passed"
