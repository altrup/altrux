#!/usr/bin/env bash
# Watches a Lambda Cloud training instance FROM THE LOCAL MACHINE and
# terminates it via the API once nothing has been training for --timeout
# seconds, so an idle instance can't silently bill. The API key stays on
# this machine -- the instance itself holds no Lambda credentials, so
# nothing running on it (including an autonomous monitoring session) can
# launch, resize, or terminate instances. It's the whole local side:
#
#   ./scripts/lambda_watchdog.sh
#
# It owns the scheduled pulling too -- lambda_pull.sh runs every
# --pull-interval seconds from the loop, so a hard crash loses at most one
# interval (lambda_pull.sh --follow exists for watchdog-less use). A session
# on the instance can also ask for a pull right now (e.g. the moment a cache
# finishes building) by touching the fetch file there:
#
#   touch ~/altrux/scripts/.watchdog-fetch
#
# The next probe pulls and deletes the marker; the fresh scripts/.pull-receipt
# lambda_pull.sh leaves on the instance is the success signal.
#
# Before terminating, it runs the final pull (retried twice on failure, then
# terminating regardless -- an unbounded billing leak is worse than a lost
# artifact, and a final pull that never succeeded leaves a loud
# scripts/PULL-FAILED-<timestamp> file on THIS machine), then once more with
# --with-mem-state for the large mem_state.pt (--no-pull skips both,
# --no-mem-state just the second). A graceful terminate is the only moment
# that knows a run is over, so it's the only place mem_state.pt can be rescued
# automatically. Both stages are bounded by --pull-timeout /
# --mem-state-timeout.
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
pattern="train.py|probe_recall.py|consolidation_null.py|capacity_ladder.py|dream_sleep.py|evaluation_run.py|experiments.lama_ckl.split|experiments.lama_ckl.runner"
terminate_cmd=""
pull=1
pull_interval=300
pull_timeout=900
mem_state=1
mem_state_timeout=3600
arm_after_training=0
arm_cap=90
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout) timeout="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --unreachable-timeout) unreachable_timeout="$2"; shift 2 ;;
    --pattern) pattern="$2"; shift 2 ;;
    --no-pull) pull=0; shift ;;
    --pull-interval) pull_interval="$2"; shift 2 ;;
    --pull-timeout) pull_timeout="$2"; shift 2 ;;
    --no-mem-state) mem_state=0; shift ;;
    --mem-state-timeout) mem_state_timeout="$2"; shift 2 ;;
    # Wait for --pattern to appear once before starting the idle countdown, so
    # arming the watchdog during a long setup (before training exists) can't
    # terminate the box. --arm-cap minutes bounds the wait (0 = forever) so a
    # setup that never starts training is still cleaned up.
    --arm-after-training) arm_after_training=1; shift ;;
    --arm-cap) arm_cap="$2"; shift 2 ;;
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
# contains the (bracketed) pattern text. Bracket the first char of EVERY
# |-alternative — one unbracketed branch would self-match the probe shell
# and read as permanent activity, disarming the watchdog entirely.
neutralized="$(printf '%s' "$pattern" | sed 's/\(^\||\)\([^[\\^.$|]\)/\1[\2]/g')"

# One round trip per probe: remote epoch, training yes/no, delay-file
# mtime, terminate-file mtime, fetch-file present.
probe_snippet="
  date +%s
  pgrep -f '$neutralized' >/dev/null && echo 1 || echo 0
  stat -c %Y '$remote_repo/scripts/.watchdog-delay' 2>/dev/null || echo 0
  stat -c %Y '$remote_repo/scripts/.watchdog-terminate' 2>/dev/null || echo 0
  [ -e '$remote_repo/scripts/.watchdog-fetch' ] && echo 1 || echo 0
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

run_pull() {
  timeout "$pull_timeout" "$SCRIPT_DIR/lambda_pull.sh" "$instance_ip"
}

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
    local log marker ok=0 attempt
    log="$(mktemp)"
    for attempt in 1 2 3; do
      echo "watchdog: final pull (resume-critical files), attempt ${attempt}/3..."
      if run_pull 2>&1 | tee "$log"; then
        ok=1
        break
      fi
      echo "watchdog: final pull failed or timed out after ${pull_timeout}s" >&2
      if (( attempt < 3 )); then sleep 120; fi
    done
    if (( ok == 0 )); then
      # The run's artifacts die with the instance, so the loss has to be
      # visible on this machine the same day rather than inferred later.
      marker="$SCRIPT_DIR/PULL-FAILED-$(date -u +%Y%m%dT%H%M%SZ)"
      {
        echo "final pull from $instance_ip (instance $instance_id) failed 3 times; terminated anyway"
        echo "reason for terminating: $reason"
        echo "last error:"
        tail -20 "$log"
      } > "$marker"
      echo "watchdog: FINAL PULL NEVER SUCCEEDED — wrote $marker" >&2
    fi
    rm -f "$log"
    if [[ "$mem_state" -eq 1 && "$ok" -eq 1 ]]; then
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
last_pull=0        # local epoch; 0 pulls on the first probe

if [[ "$arm_after_training" -eq 1 ]]; then
  echo "watchdog: waiting for '$pattern' to start before arming the idle countdown (cap: ${arm_cap}min, 0=forever)"
  arm_deadline=$(( $(date +%s) + arm_cap * 60 ))
  while true; do
    if timeout "$interval" "${SSH_CMD[@]}" "pgrep -f '$neutralized' >/dev/null" 2>/dev/null; then
      echo "watchdog: '$pattern' detected — arming"
      break
    fi
    if (( arm_cap > 0 && $(date +%s) >= arm_deadline )); then
      echo "watchdog: '$pattern' never started within ${arm_cap}min — arming anyway (a failed setup will now be cleaned up)"
      break
    fi
    echo "watchdog: [$(date +%H:%M:%S)] no '$pattern' yet — waiting ${interval}s (setup/data-gen in progress)"
    sleep "$interval"
  done
fi

while true; do
  # The timeout wrap is the hard staleness bound: a probe that somehow
  # still blocks (keepalives notwithstanding) is killed and counted as one
  # interval's worth of failure, never as the whole time it hung.
  if output="$(timeout "$interval" "${SSH_CMD[@]}" "$probe_snippet" 2>/dev/null)"; then
    unreachable_since=""
    { read -r remote_now; read -r training; read -r delay_mtime; read -r terminate_mtime; read -r fetch; } <<< "$output"
    watch_start="${watch_start:-$remote_now}"
    if (( terminate_mtime >= watch_start )); then
      terminate "termination requested via .watchdog-terminate (touched $(( remote_now - terminate_mtime ))s ago)"
    fi
    probe_now="$(date +%s)"
    if [[ "$pull" -eq 1 ]] && (( fetch == 1 || probe_now - last_pull >= pull_interval )); then
      if (( fetch == 1 )); then
        echo "watchdog: [$(date +%H:%M:%S)] pull requested via .watchdog-fetch"
      fi
      echo "watchdog: [$(date +%H:%M:%S)] pulling artifacts..."
      run_pull || echo "watchdog: pull failed or timed out after ${pull_timeout}s — retrying in ${pull_interval}s" >&2
      last_pull="$(date +%s)"
      if (( fetch == 1 )); then
        # Deleted whether or not the pull worked: the request is consumed, and
        # the receipt lambda_pull.sh leaves behind is what says it succeeded.
        timeout 30 "${SSH_CMD[@]}" "rm -f '$remote_repo/scripts/.watchdog-fetch'" 2>/dev/null || true
      fi
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
