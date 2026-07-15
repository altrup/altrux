#!/usr/bin/env bash
# Pulls training artifacts (sft/logs/ and models/*/checkpoints/) down from a
# running Lambda Cloud instance to this machine, via rsync over ssh.
#
# Run from the LOCAL machine, before terminating the instance — the run's
# logs live only on the instance and die with it:
#
#   ./scripts/lambda_pull.sh                      # one-shot; IP via the API
#   ./scripts/lambda_pull.sh 1.2.3.4              # one-shot; IP explicit
#   ./scripts/lambda_pull.sh --follow [1.2.3.4]   # re-pull every 5 minutes
#   ./scripts/lambda_pull.sh --follow --interval 60
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

remote_repo="${LAMBDA_REMOTE_REPO:-altrux}"
follow=0
interval=300
ip=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --follow) follow=1; shift ;;
    --interval) interval="$2"; shift 2 ;;
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
RSYNC=(rsync -az --info=stats1 -e "ssh -o ConnectTimeout=10 -o BatchMode=yes")

pull() {
  echo "Pulling sft/logs/ ..."
  "${RSYNC[@]}" "ubuntu@${ip}:${remote_repo}/sft/logs/" "$REPO_ROOT/sft/logs/" || return 1
  echo "Pulling models/*/checkpoints/ ..."
  "${RSYNC[@]}" --relative "ubuntu@${ip}:${remote_repo}/./models/*/checkpoints/" "$REPO_ROOT/" || return 1
}

if [[ "$follow" -eq 0 ]]; then
  pull
  echo "Done. Terminate the instance with scripts/lambda_terminate.sh (or set LAMBDA_INSTANCE_ID and run it from here)."
  exit 0
fi

echo "Following: pulling every ${interval}s until the instance stops answering (Ctrl-C to stop)..."
succeeded=0
while true; do
  if ! pull; then
    if [[ "$succeeded" -eq 0 ]]; then
      echo "error: first pull failed — check the IP, ssh access, and LAMBDA_REMOTE_REPO before trusting --follow" >&2
      exit 1
    fi
    echo "Instance stopped answering — assuming it was terminated. Last successful pull stands."
    exit 0
  fi
  succeeded=1
  echo "[$(date +%H:%M:%S)] pull ok — next in ${interval}s"
  sleep "$interval"
done
