#!/usr/bin/env bash
# Reports which instance lambda_terminate.sh would terminate, without
# terminating it — a dry run of that script's instance resolution, so a
# misconfigured setup surfaces before the terminate matters.
#
# Resolves exactly as lambda_terminate.sh does: LAMBDA_INSTANCE_ID from
# scripts/.env if set, otherwise the instance whose public IP matches this
# machine's (which only works when run from inside the instance).
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

if ! instances="$(curl -sf --max-time 10 -u "${LAMBDA_API_KEY}:" \
  https://cloud.lambdalabs.com/api/v1/instances)"; then
  echo "error: GET /instances failed — check LAMBDA_API_KEY with ./scripts/lambda_check_key.sh" >&2
  exit 1
fi

if [[ -n "${LAMBDA_INSTANCE_ID:-}" ]]; then
  echo "LAMBDA_INSTANCE_ID is set — lambda_terminate.sh would terminate $LAMBDA_INSTANCE_ID directly."
  match_by="id"
  match_value="$LAMBDA_INSTANCE_ID"
else
  echo "LAMBDA_INSTANCE_ID not set — resolving by this machine's public IP..."
  my_ip="$(curl -s --max-time 10 https://api.ipify.org)"
  if [[ -z "$my_ip" ]]; then
    echo "error: could not determine this machine's public IP" >&2
    exit 1
  fi
  echo "public IP: $my_ip"
  match_by="ip"
  match_value="$my_ip"
fi

echo "$instances" | match_by="$match_by" match_value="$match_value" python3 -c "
import json, os, sys

data = json.load(sys.stdin)
instances = data.get('data', [])
match_by = os.environ['match_by']
match_value = os.environ['match_value']

print(f'{len(instances)} instance(s) visible to this API key:')
for inst in instances:
    print(f\"  {inst.get('id')}  {inst.get('ip')}  {inst.get('status')}  {inst.get('instance_type', {}).get('name')}\")

for inst in instances:
    if inst.get(match_by) == match_value:
        print()
        print(f\"MATCH: lambda_terminate.sh would terminate {inst.get('id')} ({inst.get('ip')}, status {inst.get('status')}).\")
        sys.exit(0)

print()
if match_by == 'ip':
    print(f'NO MATCH: no instance has public IP {match_value} — lambda_terminate.sh would fail here.', file=sys.stderr)
    print('Run it from inside the instance, or set LAMBDA_INSTANCE_ID in scripts/.env.', file=sys.stderr)
else:
    print(f'NO MATCH: LAMBDA_INSTANCE_ID={match_value} is not among the instances above.', file=sys.stderr)
    print('The terminate call would be sent anyway and rejected by the API.', file=sys.stderr)
sys.exit(1)
"
