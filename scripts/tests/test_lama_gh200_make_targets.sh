#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

smoke="$(make -n -C "$ROOT/sft" lama-ckl-upstream-smoke)"
grep -Fq 'prepare-single-gpu-smoke' <<< "$smoke"
grep -Fq 'adapt-single-gpu' <<< "$smoke"
grep -Fq -- '--smoke' <<< "$smoke"
grep -Fq 'HF_HOME=' <<< "$(grep 'cd .*TAALM.*bash' <<< "$smoke")"
grep -Fq 'tee logs/lama-ckl-upstream-smoke-' <<< "$smoke"

full="$(make -n -C "$ROOT/sft" lama-ckl-upstream-run)"
grep -Fq 'adapt-single-gpu' <<< "$full"
if grep -Fq -- '--smoke' <<< "$full"; then
  echo "full upstream run uses smoke inputs" >&2
  exit 1
fi
grep -Fq 'HF_HOME=' <<< "$(grep 'cd .*TAALM.*bash' <<< "$full")"
grep -Fq 'tee logs/lama-ckl-upstream-run-' <<< "$full"

echo "LAMA GH200 Make target tests passed"
