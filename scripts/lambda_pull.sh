#!/usr/bin/env bash
# Pulls training artifacts down from a running Lambda Cloud instance to this
# machine, via rsync over ssh: sft/logs/, every models/*/checkpoints/,
# notes/ (the experimenter session's observations; the box also commits
# them to its box/<UTC> branch, this copy is the early local mirror), the
# sft/data/ artifacts listed in lambda_data_artifacts.sh,
# and the .cache/ paths listed in lambda_cache_artifacts.sh
# (so a dataset generated on the box is archived here and uploaded again on
# the next launch instead of regenerated). Whichever of those don't exist
# yet are skipped. A missing repo
# is an error in one-shot mode; in --follow mode it's expected at first
# (launch starts the pull loop before the repo upload lands) and just retried.
#
# Run from the LOCAL machine, before terminating the instance — the run's
# logs live only on the instance and die with it:
#
#   ./scripts/lambda_pull.sh                      # one-shot; IP via the API
#   ./scripts/lambda_pull.sh 1.2.3.4              # one-shot; IP explicit
#   ./scripts/lambda_pull.sh --follow [1.2.3.4]   # re-pull every 5 minutes
#   ./scripts/lambda_pull.sh --follow --interval 60
#   ./scripts/lambda_pull.sh --with-mem-state     # final pull, before terminating
#   ./scripts/lambda_pull.sh --dry-run            # list what would transfer
#
# After every successful pull, a receipt (timestamp + the sizes of the pulled
# files as they exist HERE) is written back to <remote_repo>/scripts/.pull-receipt
# on the instance: a session on the box has no other way to know its artifacts
# arrived. A failed receipt write is a warning, not a failed pull.
#
# Every pull is idempotent (rsync only moves what changed), so running it
# again — e.g. the moment data generation or filtering finishes, before the
# long training phase — costs nothing and gets the artifacts to safety early.
#
# mem_state.pt (a checkpoint's full internal model state, written by
# sft/train.py) is excluded unless --with-mem-state is passed: it's large
# enough that copying it every interval would not finish between pulls, and
# a checkpoint without it still resumes (just restarting mid-example slots).
# Pass --with-mem-state for one last pull before terminating, when exact
# mid-example resume is worth the transfer.
#
# --follow keeps pulling until the instance stops answering (i.e. it was
# terminated) or Ctrl-C, so even a hard crash mid-run loses at most one
# interval's worth of logs. Pair it with a delayed terminate on the
# instance ("; sleep 600 ;" between training and lambda_terminate.sh) so
# the final pull is guaranteed a window after training ends.
#
# Reads LAMBDA_API_KEY from scripts/.env (see scripts/.env.example) for the
# IP lookup; assumes the repo lives at ~/altrux on the instance (override
# with LAMBDA_REMOTE_REPO in scripts/.env) and ssh access as ubuntu@.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lambda_data_artifacts.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lambda_cache_artifacts.sh"

remote_repo="${LAMBDA_REMOTE_REPO:-altrux}"
follow=0
interval=300
with_mem_state=0
dry_run=0
ip=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --follow) follow=1; shift ;;
    --interval) interval="$2"; shift 2 ;;
    --with-mem-state) with_mem_state=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -*) echo "error: unknown flag $1" >&2; exit 1 ;;
    *) ip="$1"; shift ;;
  esac
done

if [[ -z "$ip" ]]; then
  if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
    echo "error: pass the instance IP as an argument, or set LAMBDA_API_KEY in scripts/.env for automatic lookup" >&2
    exit 1
  fi
  echo "Looking up running instance IP via the Lambda Cloud API..."
  ip="$(curl -sf -u "${LAMBDA_API_KEY}:" https://cloud.lambdalabs.com/api/v1/instances \
    | python3 -c "
import json, sys
instances = [i for i in json.load(sys.stdin).get('data', []) if i.get('status') == 'active']
if len(instances) != 1:
    sys.exit(f'expected exactly 1 active instance, found {len(instances)} -- pass the IP explicitly')
print(instances[0]['ip'])
")"
  echo "Found instance at $ip"
fi

# BatchMode forbids interactive auth prompts (which would hang --follow
# unattended); ConnectTimeout makes a terminated instance fail in seconds
# instead of a full TCP timeout.
SSH_OPTS=(-o ConnectTimeout=10 -o BatchMode=yes)
# .step-N.partial is a checkpoint mid-save (see sft/train.py's
# save_checkpoint); it's renamed into place when complete, so pulling it would
# just spend bandwidth on bytes that arrive again under their real name.
# -i itemizes what actually changed, so each pull's output names the files it
# brought down rather than just counting them.
RSYNC=(rsync -azi --info=stats1,progress2 --exclude='.*.partial' -e "ssh ${SSH_OPTS[*]}")
if [[ "$with_mem_state" -eq 0 ]]; then
  RSYNC+=(--exclude=mem_state.pt)
else
  echo "Including mem_state.pt (--with-mem-state) — this can be a large transfer."
fi
if [[ "$dry_run" -eq 1 ]]; then
  RSYNC+=(-n)
  echo "Dry run (--dry-run) — listing what would transfer, copying nothing."
fi

# Lists the artifact dirs that exist on the instance, one per line. Exits 10
# if the repo itself is missing, distinguishing a misconfigured
# LAMBDA_REMOTE_REPO from an artifact dir a run hasn't created yet.
artifact_dirs="sft/logs notes models/*/checkpoints"
probe_dirs="$artifact_dirs"
[[ -n "$DATA_ARTIFACTS" ]] && probe_dirs="$probe_dirs sft/data"
probe_paths() {
  ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "
    cd '$remote_repo' 2>/dev/null || exit 10
    for p in $probe_dirs; do
      [ -d \"\$p\" ] && printf '%s\n' \"\$p\"
    done
    for pat in $CACHE_ARTIFACTS; do
      for p in .cache/\$pat; do
        [ -e \"\$p\" ] && printf '%s\n' \"\$p\"
      done
    done
    exit 0
  "
}

# The instance can't see this machine's disk, so after every pull the sizes of
# the artifacts AS THEY LANDED HERE are written back to
# <remote_repo>/scripts/.pull-receipt — the only evidence a session on the box
# has that what it produced is actually home.
write_receipt() {
  {
    printf '# pull receipt %s UTC from %s (%s)\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%S)" "$(uname -n)" "$REPO_ROOT"
    cd "$REPO_ROOT" || exit
    find $artifact_dirs -type f -printf '%s %p\n' 2>/dev/null
    for pat in $DATA_ARTIFACTS; do
      stat -c '%s %n' sft/data/$pat 2>/dev/null
    done
    for pat in $CACHE_ARTIFACTS; do
      for p in .cache/$pat; do
        if [[ -d "$p" ]]; then find "$p" -type f -printf '%s %p\n'; else stat -c '%s %n' "$p" 2>/dev/null; fi
      done
    done
  } | ssh "${SSH_OPTS[@]}" "ubuntu@${ip}" "cat > '$remote_repo/scripts/.pull-receipt'"
}

# Returns 10 if the repo is missing on the instance, 1 on any other failure.
pull() {
  local paths rc=0
  paths="$(probe_paths)" || rc=$?
  if [[ "$rc" -eq 10 ]]; then
    return 10
  elif [[ "$rc" -ne 0 ]]; then
    return 1
  fi

  if [[ -z "$paths" ]]; then
    echo "Nothing to pull yet — none of $probe_dirs exist on the instance."
    return 0
  fi

  while IFS= read -r p; do
    echo "[$(date +%H:%M:%S)] Pulling $p ..."
    # sft/data holds scratch and raw intermediates alongside the artifacts
    # worth keeping, so it's the one path that pulls a filtered subset.
    filter=()
    [[ "$p" == sft/data ]] && filter=("${data_filter[@]}")
    "${RSYNC[@]}" "${filter[@]}" --relative "ubuntu@${ip}:${remote_repo}/./${p}" "$REPO_ROOT/" || return 1
  done <<< "$paths"

  if [[ "$dry_run" -eq 0 ]]; then
    write_receipt || echo "warning: the pull succeeded but writing scripts/.pull-receipt back to the instance failed" >&2
  fi
}

no_repo_msg="no repo at ~/${remote_repo} on ${ip} — set LAMBDA_REMOTE_REPO in scripts/.env if it lives elsewhere"

if [[ "$follow" -eq 0 ]]; then
  rc=0; pull || rc=$?
  if [[ "$rc" -eq 10 ]]; then
    echo "error: $no_repo_msg" >&2
    exit 1
  elif [[ "$rc" -ne 0 ]]; then
    exit 1
  fi
  echo "Done. Terminate the instance with scripts/lambda_terminate.sh (or set LAMBDA_INSTANCE_ID and run it from here)."
  exit 0
fi

echo "Following: pulling every ${interval}s until the instance stops answering (Ctrl-C to stop)..."
succeeded=0
while true; do
  rc=0; pull || rc=$?
  if [[ "$rc" -eq 10 ]]; then
    # Expected right after launch: --follow starts before the repo upload has
    # landed. Keep waiting — but if this never clears, LAMBDA_REMOTE_REPO is wrong.
    echo "[$(date +%H:%M:%S)] $no_repo_msg (expected while the repo is still uploading) — retrying in ${interval}s"
  elif [[ "$rc" -ne 0 ]]; then
    if [[ "$succeeded" -eq 0 ]]; then
      echo "error: first pull failed — check the IP and ssh access before trusting --follow" >&2
      exit 1
    fi
    echo "Instance stopped answering — assuming it was terminated. Last successful pull stands."
    exit 0
  else
    succeeded=1
    echo "[$(date +%H:%M:%S)] pull ok — next in ${interval}s"
  fi
  sleep "$interval"
done
