#!/usr/bin/env bash
# Provisions a Lambda Cloud GPU instance and hands it off to lambda_setup.sh.
#
# Runs on the LOCAL machine (it needs LAMBDA_API_KEY, which by design never
# lives on the instance). It launches the instance, waits for it to boot and
# accept ssh, then scp's lambda_setup.sh up and runs it — leaving a box that
# just needs `claude` auth and `/experimenter` to start the run.
#
# Config (scripts/.env):
#   LAMBDA_API_KEY         required
#   LAMBDA_INSTANCE_TYPE   required, e.g. gpu_1x_h100_pcie (see the Lambda UI)
#   LAMBDA_SSH_KEY_NAME    required, name of an ssh key registered in Lambda
#   LAMBDA_REGION          optional, e.g. us-east-1; auto-picks an available
#                          region for the instance type if unset
#   LAMBDA_SSH_KEY_PATH    optional, private key for ssh (default ~/.ssh/id_ed25519)
#   LAMBDA_SSH_USER        optional, default ubuntu
#   LAMBDA_INSTANCE_NAME   optional, default altrux-train
#   LAMBDA_CAPACITY_POLL_INTERVAL  optional, default 60 (seconds between
#                          capacity retries when the instance type is sold out)
#   LAMBDA_CAPACITY_MAX_WAIT  optional, default 0 = poll forever; else give up
#                          after this many seconds with no capacity
# plus any LAMBDA_SETUP_* / LAMBDA_REPO_* vars, forwarded to lambda_setup.sh.
#
#   --dry-run   resolve region + print the launch payload, then stop
#   --no-setup  launch and wait for ssh, but don't run lambda_setup.sh
#   --no-watch  don't auto-start the local watchdog + pull tmux
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="https://cloud.lambdalabs.com/api/v1"

DRY_RUN=0
RUN_SETUP=1
RUN_WATCH=1
for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=1 ;;
    --no-setup) RUN_SETUP=0 ;;
    --no-watch) RUN_WATCH=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

require() { [[ -n "${!1:-}" ]] || { echo "error: $1 not set (scripts/.env — see scripts/.env.example)" >&2; exit 1; }; }
require LAMBDA_API_KEY
require LAMBDA_INSTANCE_TYPE
require LAMBDA_SSH_KEY_NAME

SSH_USER="${LAMBDA_SSH_USER:-ubuntu}"
SSH_KEY_PATH="${LAMBDA_SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
INSTANCE_NAME="${LAMBDA_INSTANCE_NAME:-altrux-train}"
SESSION=train
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i "$SSH_KEY_PATH")

# Resolve a relative resume-checkpoint path against the repo root, so it works
# regardless of the directory launch is invoked from.
if [[ -n "${LAMBDA_RESUME_CHECKPOINT:-}" && "$LAMBDA_RESUME_CHECKPOINT" != /* ]]; then
  LAMBDA_RESUME_CHECKPOINT="$(cd "$SCRIPT_DIR/.." && pwd)/$LAMBDA_RESUME_CHECKPOINT"
fi

notify() {
  # Best-effort desktop popup (notify-send, if a desktop session is present),
  # always followed by a terminal bell — so a long unattended wait still pings.
  command -v notify-send >/dev/null 2>&1 && notify-send "lambda_launch" "$1" 2>/dev/null || true
  printf '\a' >&2
}

resolve_region() {
  # stdout: a region name with capacity (empty if none). exit 3: unknown type.
  curl -sf -u "${LAMBDA_API_KEY}:" "$API/instance-types" \
    | itype="$LAMBDA_INSTANCE_TYPE" want="${LAMBDA_REGION:-}" python3 -c "
import json, os, sys
data = json.load(sys.stdin).get('data', {})
itype, want = os.environ['itype'], os.environ['want']
info = data.get(itype)
if info is None:
    sys.stderr.write(f'unknown instance type: {itype}\n'); sys.exit(3)
regions = [r['name'] for r in info.get('regions_with_capacity_available', [])]
if want:
    print(want if want in regions else '')
else:
    print(regions[0] if regions else '')
"
}

# resolve_region exits 3 on an unknown instance type (fatal); any other
# nonzero (curl/network blip) or an empty result means "no capacity yet".
try_region() { region="$(resolve_region)" || { [[ $? == 3 ]] && exit 1; region=""; }; }

launch_instance() {
  # One launch attempt in $region. Success: sets instance_id, returns 0.
  # Capacity vanished since the check (the launch race): returns 1 to re-poll.
  # Any other API error (bad ssh key, auth, quota): prints it and exits.
  local resp rc
  resp="$(curl -s -u "${LAMBDA_API_KEY}:" "$API/instance-operations/launch" \
    -H "Content-Type: application/json" \
    -d "{\"region_name\":\"$region\",\"instance_type_name\":\"$LAMBDA_INSTANCE_TYPE\",\"ssh_key_names\":[\"$LAMBDA_SSH_KEY_NAME\"],\"name\":\"$INSTANCE_NAME\"}")"
  instance_id="$(resp="$resp" python3 -c "
import json, os, sys
raw = os.environ['resp']
try:
    d = json.loads(raw)
except json.JSONDecodeError:
    sys.stderr.write('non-JSON response from launch API:\n' + raw[:500] + '\n'); sys.exit(2)
err = d.get('error')
if err:
    code, msg = err.get('code', ''), str(err.get('message', err))
    if 'not-available' in code or 'capacity' in msg.lower():
        sys.exit(1)
    sys.stderr.write('launch failed: ' + code + ': ' + msg + '\n'); sys.exit(2)
ids = d.get('data', {}).get('instance_ids') or []
if not ids:
    sys.stderr.write('launch returned no instance_ids:\n' + raw[:500] + '\n'); sys.exit(2)
print(ids[0])
")" || { rc=$?; [[ "$rc" == 1 ]] && return 1; exit 1; }
  return 0
}

poll_interval="${LAMBDA_CAPACITY_POLL_INTERVAL:-60}"
max_wait="${LAMBDA_CAPACITY_MAX_WAIT:-0}"   # seconds to keep polling; 0 = forever

echo "Resolving a region with capacity for $LAMBDA_INSTANCE_TYPE${LAMBDA_REGION:+ in $LAMBDA_REGION}..."
try_region

if [[ "$DRY_RUN" == 1 ]]; then
  [[ -n "$region" ]] || { echo "[dry-run] no capacity for $LAMBDA_INSTANCE_TYPE${LAMBDA_REGION:+ in $LAMBDA_REGION} right now"; exit 1; }
  echo "[dry-run] would launch: type=$LAMBDA_INSTANCE_TYPE region=$region key=$LAMBDA_SSH_KEY_NAME name=$INSTANCE_NAME"
  exit 0
fi

start_ts="$(date +%s)"; polled=0
while :; do
  if [[ -z "$region" ]]; then
    polled=1
    waited=$(( $(date +%s) - start_ts ))
    if (( max_wait > 0 && waited >= max_wait )); then
      notify "gave up: no $LAMBDA_INSTANCE_TYPE capacity after ${waited}s"
      echo "error: no capacity for $LAMBDA_INSTANCE_TYPE${LAMBDA_REGION:+ in $LAMBDA_REGION} after ${waited}s" >&2
      exit 1
    fi
    echo "no capacity yet (${waited}s elapsed) — retrying in ${poll_interval}s (Ctrl-C to stop)"
    sleep "$poll_interval"
    try_region
    continue
  fi
  echo "Region: $region"
  echo "Launching..."
  launch_instance && break
  # The launch lost the capacity race — scarce types can sell out in the
  # seconds between the capacity check and the launch call. Resume polling.
  echo "capacity in $region vanished before launch — back to polling"
  region=""
done
[[ "$polled" == 1 ]] && notify "$LAMBDA_INSTANCE_TYPE launched in $region"
echo "Instance: $instance_id"

echo -n "Waiting for boot"
ip=""
for _ in $(seq 1 120); do
  read -r status ip < <(curl -sf -u "${LAMBDA_API_KEY}:" "$API/instances/$instance_id" \
    | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print(d.get('status',''), d.get('ip') or '')")
  [[ "$status" == "active" && -n "$ip" ]] && break
  echo -n "."
  sleep 10
done
echo
[[ -n "$ip" ]] || { notify "instance never became active (id $instance_id)"; echo "error: instance never became active — check the Lambda UI (id $instance_id)" >&2; exit 1; }
echo "Active at $ip"

echo -n "Waiting for ssh"
for _ in $(seq 1 60); do
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" true 2>/dev/null && break
  echo -n "."
  sleep 5
done
echo
notify "instance ready at $ip — setup starting"

# Local billing-protection stack: pull --follow (rescues artifacts) + watchdog
# (terminates the instance once training stops). --arm-after-training holds the
# watchdog's idle countdown until train.py first appears, so it can't kill the
# box during the minutes-long setup/data-gen before training starts. Started
# before setup so protection is live for the whole run.
if [[ "$RUN_WATCH" == 1 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "note: tmux not found locally — NOT auto-starting watchdog/pull; run scripts/lambda_watchdog.sh yourself or the box bills unbounded"
  elif tmux has-session -t altrux-watch 2>/dev/null; then
    echo "local watch tmux 'altrux-watch' already exists — leaving it"
  else
    tmux new-session -d -s altrux-watch -n pull "'$SCRIPT_DIR/lambda_pull.sh' --follow '$ip'; exec bash"
    tmux new-window -t altrux-watch -n watchdog "LAMBDA_INSTANCE_ID='$instance_id' LAMBDA_INSTANCE_IP='$ip' '$SCRIPT_DIR/lambda_watchdog.sh' --arm-after-training; exec bash"
    echo "Billing protection up in tmux 'altrux-watch' (pull + watchdog). Attach: tmux attach -t altrux-watch"
  fi
fi

if [[ "$RUN_SETUP" == 0 ]]; then
  echo "Skipping setup (--no-setup). Finish with:"
  echo "  scp -i $SSH_KEY_PATH $SCRIPT_DIR/lambda_setup.sh $SSH_USER@$ip:"
  echo "  ssh -i $SSH_KEY_PATH $SSH_USER@$ip 'bash lambda_setup.sh'"
  exit 0
fi

if [[ -n "${LAMBDA_RESUME_CHECKPOINT:-}" ]]; then
  [[ -d "$LAMBDA_RESUME_CHECKPOINT" ]] || { echo "error: LAMBDA_RESUME_CHECKPOINT is not a directory: $LAMBDA_RESUME_CHECKPOINT" >&2; exit 1; }
  export LAMBDA_RESUME_EPOCH="$(basename "$(dirname "$LAMBDA_RESUME_CHECKPOINT")")"
  echo "Uploading resume checkpoint $(basename "$LAMBDA_RESUME_CHECKPOINT") ($(du -sh "$LAMBDA_RESUME_CHECKPOINT" | cut -f1)) to staging..."
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "rm -rf ~/resume-staging && mkdir -p ~/resume-staging"
  scp -r "${SSH_OPTS[@]}" "$LAMBDA_RESUME_CHECKPOINT" "$SSH_USER@$ip:resume-staging/"
fi

# Global Claude config, so the instance's claude behaves like the local one
# (global CLAUDE.md, status line, skills). settings.json deliberately stays
# local: its deny rules (git push) would block the experimenter, and setup
# merges the statusLine entry into the instance's own settings instead.
# Credentials travel separately via CLAUDE_CODE_OAUTH_TOKEN.
claude_files=()
for f in CLAUDE.md statusline.sh keybindings.json skills commands agents; do
  [[ -e "$HOME/.claude/$f" ]] && claude_files+=(".claude/$f")
done
if (( ${#claude_files[@]} )); then
  echo "Uploading global Claude config (${claude_files[*]#.claude/})..."
  tar -C "$HOME" -czf - "${claude_files[@]}" | ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "tar -xzf - -C ~"
fi

echo "Uploading setup + config..."
scp "${SSH_OPTS[@]}" "$SCRIPT_DIR/lambda_setup.sh" "$SSH_USER@$ip:lambda_setup.sh"

# Forward config (incl. secrets) via a file rather than the command line, so
# tokens don't land in the instance's process list. Removed after setup reads it.
env_file="$(mktemp)"
trap 'rm -f "$env_file"' EXIT
for v in LAMBDA_REPO_URL LAMBDA_REMOTE_REPO LAMBDA_REPO_REF LAMBDA_MODEL_NAME \
         LAMBDA_SETUP_DATA_CMD LAMBDA_SETUP_DATA_MARKER LAMBDA_RESUME_EPOCH \
         GITHUB_TOKEN HF_TOKEN CLAUDE_CODE_OAUTH_TOKEN TORCH_BACKEND MAX_JOBS; do
  [[ -n "${!v:-}" ]] && printf 'export %s=%q\n' "$v" "${!v}" >> "$env_file"
done
scp "${SSH_OPTS[@]}" "$env_file" "$SSH_USER@$ip:lambda_setup.env"

echo "Running setup in tmux session '$SESSION' (attaching live)."
echo "Detach with Ctrl-b d; reattach later: ssh -i $SSH_KEY_PATH $SSH_USER@$ip -t tmux attach -t $SESSION"
ssh -t "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
  "command -v tmux >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq tmux; }; tmux new-session -A -s $SESSION 'set -a; . ~/lambda_setup.env 2>/dev/null; set +a; rm -f ~/lambda_setup.env; bash lambda_setup.sh; exec bash -l'"

echo
echo "Instance $instance_id is at $ip."
echo "Two tmux sessions on it: '$SESSION' (training runs here) and 'experimenter'"
echo "(attach with 'tmux attach -t experimenter', run claude, then /experimenter)."
