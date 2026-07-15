#!/usr/bin/env bash
# Terminates this Lambda instance (via lambda_terminate.sh) once NOTHING has
# been training for --timeout seconds, so an idle instance can't silently
# bill. Run it on the instance, in the background, alongside the run:
#
#   nohup ./scripts/lambda_watchdog.sh >> watchdog.log 2>&1 &
#
# "Training" means a process matching --pattern (default: train.py) exists.
# Anyone working on the instance (e.g. a Claude Code session between runs)
# can DELAY termination by touching the delay file:
#
#   touch scripts/.watchdog-delay
#
# A touch grants at most one --timeout window from the moment of the touch —
# the deadline is max(last training activity, delay-file mtime) + timeout,
# and future-dated mtimes are clamped to now — so the timer can be pushed
# back indefinitely only by touching it again every <timeout seconds,
# never paused outright. This is a guardrail against forgetting, not a
# security boundary: anything on the instance could also just kill this
# process.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

timeout=600
interval=30
pattern="train.py"
delay_file="$SCRIPT_DIR/.watchdog-delay"
terminate_cmd="$SCRIPT_DIR/lambda_terminate.sh"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) timeout="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --pattern) pattern="$2"; shift 2 ;;
    --delay-file) delay_file="$2"; shift 2 ;;
    # Override what runs on timeout -- e.g. `--terminate-cmd "echo boom"`
    # to dry-run the countdown without a real terminate.
    --terminate-cmd) terminate_cmd="$2"; shift 2 ;;
    *) echo "error: unknown flag $1" >&2; exit 1 ;;
  esac
done

training_running() {
  # Exclude our own ancestry and anything mentioning this script: a
  # --pattern passed on the command line appears verbatim in the shell
  # chain that launched us, and matching ourselves would hold the
  # instance open forever.
  local pids exclude p pid
  pids="$(pgrep -f "$pattern" || true)"
  [[ -z "$pids" ]] && return 1
  exclude=" $$ "
  p="$PPID"
  while [[ -n "$p" && "$p" -gt 1 ]]; do
    exclude+="$p "
    p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
  done
  for pid in $pids; do
    [[ "$exclude" == *" $pid "* ]] && continue
    # Already gone -- e.g. our own $(pgrep) command-substitution fork,
    # which inherits this script's argv (pattern included) but dies
    # within the same check.
    [[ -d "/proc/$pid" ]] || continue
    grep -aq "lambda_watchdog" "/proc/$pid/cmdline" 2>/dev/null && continue
    return 0
  done
  return 1
}

echo "watchdog: terminating after ${timeout}s without training (pattern: '$pattern') or a touch of $delay_file"
last_active="$(date +%s)"  # startup counts as activity: a full grace window to get training going

while true; do
  now="$(date +%s)"
  if training_running; then
    last_active="$now"
  fi

  delay_mtime=0
  if [[ -f "$delay_file" ]]; then
    delay_mtime="$(stat -c %Y "$delay_file")"
    if (( delay_mtime > now )); then
      # A future-dated mtime would hold the deadline open forever; rewrite
      # it to now so it grants exactly one window, like any other touch.
      touch "$delay_file"
      delay_mtime="$now"
    fi
  fi

  effective=$(( last_active > delay_mtime ? last_active : delay_mtime ))
  remaining=$(( effective + timeout - now ))

  if (( remaining <= 0 )); then
    echo "watchdog: no training and no delay touch for ${timeout}s — terminating instance"
    exec bash -c "$terminate_cmd"
  fi

  if ! training_running; then
    echo "watchdog: [$(date +%H:%M:%S)] nothing training — terminating in ${remaining}s unless training resumes or $delay_file is touched"
  fi
  sleep "$interval"
done
