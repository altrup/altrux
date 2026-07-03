#!/usr/bin/env bash
# Terminates the current Lambda Cloud instance via the API.
#
# A Lambda instance's own terminal can only shut down its OS, not stop billing —
# that requires calling the Lambda Cloud API's terminate endpoint, which this
# script wraps so it can be chained after a training run, e.g.:
#
#   make train && ./scripts/lambda_terminate.sh
#
# Reads LAMBDA_API_KEY from scripts/.env (see scripts/.env.example). If
# LAMBDA_INSTANCE_ID is also set there, that instance is terminated directly;
# otherwise the script looks up the running instance's ID by matching this
# machine's public IP against `GET /instances`, which only works when run from
# inside the instance itself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

if [[ -z "${LAMBDA_API_KEY:-}" ]]; then
  echo "error: LAMBDA_API_KEY not set (add it to scripts/.env — see scripts/.env.example)" >&2
  exit 1
fi

instance_id="${LAMBDA_INSTANCE_ID:-}"

if [[ -z "$instance_id" ]]; then
  echo "LAMBDA_INSTANCE_ID not set — looking up running instance by public IP..."
  my_ip="$(curl -s --max-time 10 https://api.ipify.org)"
  if [[ -z "$my_ip" ]]; then
    echo "error: could not determine this machine's public IP" >&2
    exit 1
  fi

  instance_id="$(curl -sf -u "${LAMBDA_API_KEY}:" https://cloud.lambdalabs.com/api/v1/instances \
    | my_ip="$my_ip" python3 -c "
import json, os, sys
data = json.load(sys.stdin)
my_ip = os.environ['my_ip']
for inst in data.get('data', []):
    if inst.get('ip') == my_ip:
        print(inst['id'])
        break
")"

  if [[ -z "$instance_id" ]]; then
    echo "error: no running instance found matching this machine's IP ($my_ip)" >&2
    echo "set LAMBDA_INSTANCE_ID in scripts/.env explicitly instead" >&2
    exit 1
  fi
fi

echo "Terminating Lambda Cloud instance $instance_id..."
curl -sf -u "${LAMBDA_API_KEY}:" \
  https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" \
  -d "{\"instance_ids\":[\"$instance_id\"]}"
echo
echo "Terminate request sent for instance $instance_id."
