#!/usr/bin/env bash
# Checks whether LAMBDA_API_KEY (scripts/.env) is a valid Lambda Cloud API key.
#
# The Lambda Cloud API has no dedicated "validate key" endpoint, so this hits
# GET /instance-types — a lightweight, always-available endpoint that requires
# auth but no running instance — and reports whether the key was accepted.
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

http_status="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
  -u "${LAMBDA_API_KEY}:" \
  https://cloud.lambdalabs.com/api/v1/instance-types)"

case "$http_status" in
  200)
    echo "LAMBDA_API_KEY is valid."
    ;;
  401)
    echo "LAMBDA_API_KEY is invalid or unauthorized (HTTP 401)." >&2
    exit 1
    ;;
  *)
    echo "error: unexpected HTTP status $http_status from Lambda Cloud API" >&2
    exit 1
    ;;
esac
