#!/usr/bin/env bash
# The watchdog reads liveness from the delay file's mtime alone; probes run in
# a local shell (WATCHDOG_SSH_OVERRIDE) against a fake remote repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$work"' EXIT
# A copy, so the real scripts/.env and lock are never read.
mkdir -p "$work/local" "$work/remote/scripts"
cp "$ROOT/scripts/lambda_watchdog.sh" "$work/local/"
delay="$work/remote/scripts/.watchdog-delay"
fail() { echo "FAIL: $*" >&2; exit 1; }

watchdog() {
  LAMBDA_INSTANCE_ID=test LAMBDA_INSTANCE_IP=127.0.0.1 LAMBDA_REMOTE_REPO="$work/remote" \
    WATCHDOG_SSH_OVERRIDE="bash -c" timeout 30 "$work/local/lambda_watchdog.sh" \
    --interval 1 --timeout 3 --no-pull --terminate-cmd "echo TERMINATED" "$@"
}
# Prints the seconds the watchdog ran before terminating.
timed() {
  local start=$SECONDS out
  out="$(watchdog "$@" 2>&1)" || fail "watchdog exited nonzero: $out"
  grep -q TERMINATED <<< "$out" || fail "never terminated: $out"
  echo $(( SECONDS - start ))
}
keep_touching() { for _ in $(seq "$1"); do touch "$delay"; sleep 1; done; }

# A running process is no longer evidence of life.
rc=0; watchdog --pattern sleep >/dev/null 2>&1 || rc=$?
(( rc == 1 )) || fail "--pattern still accepted (exit $rc)"
sleep 30 & sleep_pid=$!
rm -f "$delay"
(( $(timed) < 8 )) || fail "a live process kept an untouched box alive"
kill "$sleep_pid"

# Touches keep it alive; it terminates one timeout after they stop.
keep_touching 6 &
(( $(timed) >= 7 )) || fail "terminated while the delay file was being touched"
wait

# A future-dated touch grants one window, not forever.
touch -d '+1 hour' "$delay"
(( $(timed) < 8 )) || fail "a future mtime held the box open"
(( $(stat -c %Y "$delay") <= $(date +%s) )) || fail "future mtime not rewritten"

# --arm-after-training waits for the first touch, and a stale file is not one.
touch -d '-1 hour' "$delay"
( sleep 5; keep_touching 2 ) &
out="$(watchdog --arm-after-training --arm-cap 0 2>&1)" || true
grep -q 'arming$' <<< "$out" || fail "never armed on the touch: $out"
grep -c 'no delay-file touch yet' <<< "$out" | { read -r n; (( n >= 3 )); } || fail "armed on the stale file: $out"
wait

if grep -q 'lambda_watchdog.sh.*--pattern' "$ROOT/scripts/lambda_launch.sh"; then
  fail "launch passes the removed --pattern flag"
fi

echo "lambda watchdog delay-file tests passed"
