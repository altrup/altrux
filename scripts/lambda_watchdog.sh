#!/usr/bin/env bash
# Watches a Lambda Cloud training instance FROM THE LOCAL MACHINE and
# terminates it via the API once nothing has been training for --timeout
# seconds, so an idle instance can't silently bill. The API key stays on
# this machine -- the instance itself holds no Lambda credentials, so
# nothing running on it (including an autonomous monitoring session) can
# launch, resize, or terminate instances. Run it alongside the pull loop:
#
#   ./scripts/lambda_watchdog.sh &
#   ./scripts/lambda_pull.sh --follow
#
# Before terminating, it runs lambda_pull.sh twice: once for the
# resume-critical files, then once with --with-mem-state for the large
# mem_state.pt (--no-pull skips both, --no-mem-state just the second). A
# graceful terminate is the only moment that knows a run is over, so it's the
# only place mem_state.pt can be rescued automatically. Both pulls are
# best-effort and separately bounded (--pull-timeout, --mem-state-timeout) --
# terminate follows whether they succeed, fail, or time out. The --follow loop
# stays worth running alongside: it bounds what a hard crash loses, where no
# graceful terminate ever happens.
#
# "Training" means a process matching --pattern (default: train.py) exists
# on the instance, probed over ssh every --interval seconds. Anyone working
# on the instance (e.g. a Claude Code session between runs) can DELAY
# termination by touching the delay file there:
#
#   touch ~/altrux/scripts/.watchdog-delay
#
# A touch grants at most one --timeout window from the moment of the touch
# (future-dated mtimes are rewritten to now), so the timer can be pushed
# back indefinitely only by touching again every <timeout seconds, never
# paused outright.
#
# The inverse also exists: touching the terminate file on the instance
#
#   touch ~/altrux/scripts/.watchdog-terminate
#
# makes the next probe terminate immediately (with the usual final pulls)
# instead of waiting out --timeout -- how a session on the instance, which
# holds no API key, says "this run is over, stop billing now". Only touches
# made after this watchdog's first successful probe count: a stale file
# left over from a previous run (or restored by a repo sync) must not kill
# a healthy run at startup. Requesting termination is the one instance->API
# direction that's safe to allow -- it can only stop billing, never extend it.
#
# If the instance stops answering ssh while the API still reports it
# active, it terminates after --unreachable-timeout seconds anyway -- an
# instance that can't be reached can't be trained on, and shouldn't bill.
# Each probe is hard-bounded to --interval seconds (ssh keepalives plus a
# timeout wrap), so unreachability evidence is never staler than one
# interval -- a single hung ssh connection can't silently swallow many
# minutes and then vault past the threshold in one step.
# The flip side of keeping the key local: if THIS machine sleeps or loses
# network, nothing stops the billing. Keep it awake for the whole run.
#
# Reads LAMBDA_API_KEY from scripts/.env. The instance is found via the API
# (expects exactly one active instance); set LAMBDA_INSTANCE_ID and
# LAMBDA_INSTANCE_IP in scripts/.env to target one explicitly. Repo path on
# the instance defaults to ~/altrux (LAMBDA_REMOTE_REPO to override), ssh
# user to ubuntu (LAMBDA_SSH_USER to override).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

timeout=1800
interval=60
unreachable_timeout=900
pattern="train.py"
terminate_cmd=""
pull=1
pull_timeout=900
mem_state=1
mem_state_timeout=3600
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) timeout="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --unreachable-timeout) unreachable_timeout="$2"; shift 2 ;;
    --pattern) pattern="$2"; shift 2 ;;
    --no-pull) pull=0; shift ;;
    --pull-timeout) pull_timeout="$2"; shift 2 ;;
    --no-mem-state) mem_state=0; shift ;;
    --mem-state-timeout) mem_state_timeout="$2"; shift 2 ;;
    # Override what runs on timeout -- e.g. `--terminate-cmd "echo boom"`
    # to dry-run the countdown without a real terminate.
    --terminate-cmd) terminate_cmd="$2"; shift 2 ;;
    *) echo "error: unknown flag $1" >&2; exit 1 ;;
  esac
done

# Two watchdogs against one instance double the terminate risk and produce
# interleaved, misleading logs -- refuse to be the second.
exec 9>"$SCRIPT_DIR/.watchdog.lock"
if ! flock -n 9; then
  echo "error: another lambda_watchdog.sh is already running (holds $SCRIPT_DIR/.watchdog.lock)" >&2
  exit 1
fi

remote_repo="${LAMBDA_REMOTE_REPO:-altrux}"
ssh_user="${LAMBDA_SSH_USER:-ubuntu}"

instance_id="${LAMBDA_INSTANCE_ID:-}"
instance_ip="${LAMBDA_INSTANCE_IP:-}"
if [[ -z "$instance_id" || -z "$instance_ip" ]]; then
  if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
    echo "error: LAMBDA_API_KEY not set (add it to scripts/.env — see scripts/.env.example)" >&2
    exit 1
  fi
  echo "Looking up the active instance via the Lambda Cloud API..."
  read -r instance_id instance_ip < <(
    curl -sf --max-time 30 -u "${LAMBDA_API_KEY}:" https://cloud.lambdalabs.com/api/v1/instances \
      | python3 -c "
import json, sys
instances = [i for i in json.load(sys.stdin).get('data', []) if i.get('status') == 'active']
if len(instances) != 1:
    sys.exit(f'expected exactly 1 active instance, found {len(instances)} -- set LAMBDA_INSTANCE_ID/LAMBDA_INSTANCE_IP explicitly')
print(instances[0]['id'], instances[0]['ip'])
")
fi
echo "Watching instance $instance_id at $instance_ip (pattern: '$pattern', timeout: ${timeout}s)"

# Neutralize the pattern for pgrep -f on the instance: '[t]rain.py' matches
# a running train.py but not the probe shell whose own command line
# contains the (bracketed) pattern text.
neutralized="$(printf '%s' "$pattern" | sed 's/^\([^[\\^.$]\)/[\1]/')"

# One round trip per probe: remote epoch, training yes/no, delay-file
# mtime, terminate-file mtime.
probe_snippet="
  date +%s
  pgrep -f '$neutralized' >/dev/null && echo 1 || echo 0
  stat -c %Y '$remote_repo/scripts/.watchdog-delay' 2>/dev/null || echo 0
  stat -c %Y '$remote_repo/scripts/.watchdog-terminate' 2>/dev/null || echo 0
"

# ConnectTimeout only bounds the TCP connect; the keepalives kill a
# connection that establishes and then goes dead (otherwise a probe can
# block on kernel TCP timeouts for 15+ minutes).
SSH_CMD=(ssh -o ConnectTimeout=10 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=2 "${ssh_user}@${instance_ip}")
# Test hook: WATCHDOG_SSH_OVERRIDE="bash -c" runs probes in a local shell
# instead of over ssh.
if [[ -n "${WATCHDOG_SSH_OVERRIDE:-}" ]]; then
  read -ra SSH_CMD <<< "$WATCHDOG_SSH_OVERRIDE"
fi

# $2: whether to attempt a final pull first (0 on the unreachable path --
# nothing can be pulled from an instance that won't answer ssh).
terminate() {
  local reason="$1" do_pull="${2:-1}"
  echo "watchdog: $reason — terminating instance $instance_id"
  if [[ -n "$terminate_cmd" ]]; then
    exec bash -c "$terminate_cmd"
  fi
  if [[ "$pull" -eq 1 && "$do_pull" -eq 1 ]]; then
    # Two stages, small-files-first: mem_state.pt sorts before optimizer.pt /
    # state.pt / trainable.pt within a checkpoint, so a single --with-mem-state
    # pull that hits its timeout mid-transfer would starve exactly the files a
    # resume needs. Both stages are best-effort -- the terminate has to happen
    # even if a pull hangs, since an unbounded billing leak is the one thing
    # this script exists to prevent.
    echo "watchdog: final pull (resume-critical files) before terminating..."
    timeout "$pull_timeout" "$SCRIPT_DIR/lambda_pull.sh" "$instance_ip" \
      || echo "watchdog: final pull failed or timed out after ${pull_timeout}s — terminating anyway" >&2
    if [[ "$mem_state" -eq 1 ]]; then
      echo "watchdog: pulling mem_state.pt (large; --no-mem-state to skip)..."
      timeout "$mem_state_timeout" "$SCRIPT_DIR/lambda_pull.sh" --with-mem-state "$instance_ip" \
        || echo "watchdog: mem_state.pt pull failed or timed out after ${mem_state_timeout}s — terminating anyway (checkpoints still resume without it)" >&2
    fi
  fi
  LAMBDA_INSTANCE_ID="$instance_id" exec "$SCRIPT_DIR/lambda_terminate.sh"
}

instance_still_active() {
  [[ -n "${LAMBDA_API_KEY:-}" ]] || return 0  # can't check without the API; assume active
  local status
  status="$(curl -sf --max-time 30 -u "${LAMBDA_API_KEY}:" \
    "https://cloud.lambdalabs.com/api/v1/instances/${instance_id}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['status'])" 2>/dev/null)" || return 0
  [[ "$status" == "active" ]]
}

last_active=""      # in REMOTE clock terms; set by the first successful probe (grace window)
watch_start=""      # remote epoch of the first successful probe; older terminate touches are stale
unreachable_since=""

while true; do
  # The timeout wrap is the hard staleness bound: a probe that somehow
  # still blocks (keepalives notwithstanding) is killed and counted as one
  # interval's worth of failure, never as the whole time it hung.
  if output="$(timeout "$interval" "${SSH_CMD[@]}" "$probe_snippet" 2>/dev/null)"; then
    unreachable_since=""
    { read -r remote_now; read -r training; read -r delay_mtime; read -r terminate_mtime; } <<< "$output"
    watch_start="${watch_start:-$remote_now}"
    if (( terminate_mtime >= watch_start )); then
      terminate "termination requested via .watchdog-terminate (touched $(( remote_now - terminate_mtime ))s ago)"
    fi
    if [[ -z "$last_active" || "$training" == "1" ]]; then
      last_active="$remote_now"
    fi
    if (( delay_mtime > remote_now + 60 )); then
      # A future-dated mtime would hold the deadline open forever; rewrite
      # it to now so it grants exactly one window, like any other touch.
      timeout 30 "${SSH_CMD[@]}" "touch '$remote_repo/scripts/.watchdog-delay'" 2>/dev/null || true
      delay_mtime="$remote_now"
    fi
    effective=$(( last_active > delay_mtime ? last_active : delay_mtime ))
    remaining=$(( effective + timeout - remote_now ))
    if (( remaining <= 0 )); then
      terminate "no training and no delay touch for ${timeout}s"
    fi
    # Every probe logs: an unguarded instance and a healthy one must not look
    # alike, and a silent watchdog is indistinguishable from a dead one.
    ts="[$(date +%H:%M:%S)]"
    if [[ "$training" == "1" ]]; then
      echo "watchdog: $ts training ($pattern) — next check in ${interval}s"
    elif (( delay_mtime > last_active )); then
      echo "watchdog: $ts nothing training — delay touched $(( remote_now - delay_mtime ))s ago — terminating in ${remaining}s"
    else
      echo "watchdog: $ts nothing training — terminating in ${remaining}s unless training resumes or the delay file is touched"
    fi
  else
    if ! instance_still_active; then
      echo "watchdog: instance is no longer active — exiting"
      exit 0
    fi
    now="$(date +%s)"
    unreachable_since="${unreachable_since:-$now}"
    unreachable_for=$(( now - unreachable_since ))
    echo "watchdog: [$(date +%H:%M:%S)] instance unreachable for ${unreachable_for}s (API says active) — terminating at ${unreachable_timeout}s"
    if (( unreachable_for >= unreachable_timeout )); then
      terminate "unreachable for ${unreachable_timeout}s while billing" 0
    fi
  fi
  sleep "$interval"
done
