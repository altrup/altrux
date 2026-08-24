#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
watchdog="$ROOT/scripts/lambda_watchdog.sh"
launch="$ROOT/scripts/lambda_launch.sh"

grep -Fq 'evaluation_run.py' "$watchdog"
grep -Fq 'experiments.lama_ckl.split' "$watchdog"
grep -Fq 'experiments.lama_ckl.runner' "$watchdog"

if grep 'lambda_watchdog.sh.*--pattern' "$launch" >/dev/null; then
  echo "launch overrides the watchdog's authoritative default pattern" >&2
  exit 1
fi

echo "lambda LAMA watchdog tests passed"
