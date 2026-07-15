#!/usr/bin/env bash
# Pulls training artifacts (sft/logs/ and models/*/checkpoints/) down from a
# running Lambda Cloud instance to this machine, via rsync over ssh.
#
# Run from the LOCAL machine, before terminating the instance — the run's
# logs live only on the instance and die with it:
#
#   ./scripts/lambda_pull.sh            # instance IP looked up via the API
#   ./scripts/lambda_pull.sh 1.2.3.4    # or given explicitly
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
ip="${1:-}"

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

echo "Pulling sft/logs/ ..."
rsync -avz --progress "ubuntu@${ip}:${remote_repo}/sft/logs/" "$REPO_ROOT/sft/logs/"

echo "Pulling models/*/checkpoints/ ..."
rsync -avz --progress --relative "ubuntu@${ip}:${remote_repo}/./models/*/checkpoints/" "$REPO_ROOT/"

echo "Done. Terminate the instance with scripts/lambda_terminate.sh (or set LAMBDA_INSTANCE_ID and run it from here)."
