#!/usr/bin/env bash
# Copies the compiled-from-source wheels (mamba-ssm, causal-conv1d) out of a
# running instance's uv cache into .cache/wheels/ at the repo root, so the
# next lambda_launch.sh uploads them and setup installs in seconds instead of
# recompiling (~10+ min). Run from the LOCAL machine any time after the
# instance's `make sync` has finished — the wheels live only in the
# instance's cache and die with it.
#
#   ./scripts/lambda_harvest_wheels.sh            # IP via the API
#   ./scripts/lambda_harvest_wheels.sh 1.2.3.4    # IP explicit
#
# Wheels are arch/python-tagged (e.g. cp314 linux_aarch64 for a GH200 box) —
# a future box with different tags ignores them and builds from source, so
# stale wheels are harmless except after a torch version bump (not encoded in
# the filename): clear .cache/wheels/ when torch changes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WHEEL_DIR="$REPO_ROOT/.cache/wheels"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

SSH_USER="${LAMBDA_SSH_USER:-ubuntu}"
SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
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

SSH_OPTS=(-o ConnectTimeout=10 -o BatchMode=yes -i "$SSH_KEY_PATH")

wheels="$(ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
  "find ~/.cache/uv \( -name 'mamba_ssm*.whl' -o -name 'causal_conv1d*.whl' \) 2>/dev/null")"
if [[ -z "$wheels" ]]; then
  echo "no built wheels in the instance's uv cache yet — run again after 'make sync' finishes" >&2
  exit 1
fi

mkdir -p "$WHEEL_DIR"
while IFS= read -r w; do
  echo "pulling $(basename "$w")..."
  scp "${SSH_OPTS[@]}" "$SSH_USER@$ip:$w" "$WHEEL_DIR/"
done <<< "$wheels"
echo "Stashed in $WHEEL_DIR:"
ls -lh "$WHEEL_DIR"
