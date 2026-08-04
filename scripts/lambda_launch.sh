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
#   LAMBDA_DATA_ARTIFACTS  optional, sft/data/ globs to upload (see
#                          lambda_data_artifacts.sh); empty = upload no data
#   LAMBDA_CAPACITY_POLL_INTERVAL  optional, default 60 (seconds between
#                          capacity retries when the instance type is sold out)
#   LAMBDA_CAPACITY_MAX_WAIT  optional, default 0 = poll forever; else give up
#                          after this many seconds with no capacity
# plus any LAMBDA_SETUP_* / LAMBDA_REPO_* vars, forwarded to lambda_setup.sh.
#
#   --dry-run   resolve region + print the launch payload, then stop
#   --no-setup  launch and wait for ssh, but don't run lambda_setup.sh
#   --no-watch  don't auto-start the local 'altrux' tmux (pull + watchdog +
#               remote-attach windows)
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

# LAMBDA_RESUME_CHECKPOINT: space-separated checkpoint step dirs (one or
# more — several when the plan probes/evals across checkpoints), uploaded
# WITHOUT optimizer.pt (~2/3 of each checkpoint; probes never read it).
# LAMBDA_RESUME_CHECKPOINT_FULL: same format, uploaded WITH optimizer.pt —
# for the checkpoint training resumes from, avoiding the fresh-optimizer
# loss transient. A step dir listed in both gets its optimizer (FULL wins).
# Uploaded path-preserving (rsync --relative from the repo root), so each
# checkpoint lands on the instance under the same models/<model>/checkpoints/
# path it has here — the path itself records which model it belongs to, and
# checkpoints for several models can ship in one run. Paths must therefore
# live under the repo root; relative ones resolve against it.
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
resolve_ckpt() {
  # stdout: absolute path with rsync's /./ relative-anchor at the repo root.
  local p="$1"
  [[ "$p" != /* ]] && p="$ROOT/$p"
  [[ "$p" == "$ROOT"/* ]] || { echo "error: checkpoint not under repo root: $p" >&2; exit 1; }
  echo "$ROOT/./${p#"$ROOT"/}"
}
resume_ckpts=()
for p in ${LAMBDA_RESUME_CHECKPOINT:-}; do resume_ckpts+=("$(resolve_ckpt "$p")"); done
resume_ckpts_full=()
for p in ${LAMBDA_RESUME_CHECKPOINT_FULL:-}; do resume_ckpts_full+=("$(resolve_ckpt "$p")"); done
all_ckpts=("${resume_ckpts[@]}" "${resume_ckpts_full[@]}")

# Training-data artifacts already generated here (see lambda_data_artifacts.sh)
# ride up the same path-preserving staging route as checkpoints, so the box
# only ever generates what this machine doesn't already have.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lambda_data_artifacts.sh"

# Validate + confirm the run config BEFORE launching, while aborting is
# still free (no instance billing yet).
if [[ "$RUN_SETUP" == 1 && "$DRY_RUN" == 0 ]]; then
  [[ -f "$SSH_KEY_PATH" ]] || { echo "error: ssh private key not found: $SSH_KEY_PATH (LAMBDA_SSH_KEY_PATH)" >&2; exit 1; }
  for p in "${all_ckpts[@]}"; do
    [[ -d "$p" ]] || { echo "error: LAMBDA_RESUME_CHECKPOINT(_FULL) entry is not a directory: $p" >&2; exit 1; }
  done
  echo "Instance: $LAMBDA_INSTANCE_TYPE${LAMBDA_REGION:+ in $LAMBDA_REGION}"
  echo "SSH key: $LAMBDA_SSH_KEY_NAME (private key: $SSH_KEY_PATH)"
  echo "Repo ref: ${LAMBDA_REPO_REF:-(default branch)}"
  if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
    echo "Experimenter: auto-starts (CLAUDE_CODE_OAUTH_TOKEN set)"
  else
    echo "Experimenter: MANUAL — no CLAUDE_CODE_OAUTH_TOKEN; box idles (billing) until you attach and auth"
  fi
  if (( ${#all_ckpts[@]} )); then
    echo "Checkpoints to upload ($(du -shc "${all_ckpts[@]}" | tail -1 | cut -f1) before exclusions; from"
    echo "LAMBDA_RESUME_CHECKPOINT(_FULL) in scripts/.env, landing at the same repo-relative path):"
    for p in "${resume_ckpts[@]}"; do echo "  ${p#*/./}"; done
    for p in "${resume_ckpts_full[@]}"; do echo "  ${p#*/./} (with optimizer.pt)"; done
  else
    echo "Checkpoints to upload: none (fresh run)"
  fi
  if (( ${#data_local_files[@]} )); then
    echo "Data artifacts to upload ($(du -shc "${data_local_files[@]}" | tail -1 | cut -f1)) — the box regenerates only what is NOT listed:"
    for p in "${data_local_files[@]}"; do printf '  %5s  %s\n' "$(du -h "$p" | cut -f1)" "${p#*/./}"; done
  else
    echo "Data artifacts to upload: none (the box generates all of its own)"
  fi
  if [[ -t 0 ]]; then
    read -r -p "Proceed? [Y/n] " reply
    [[ "$reply" =~ ^[Nn] ]] && { echo "aborted — edit scripts/.env and relaunch"; exit 1; }
  fi
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

poll_interval="${LAMBDA_CAPACITY_POLL_INTERVAL:-30}"
burst_attempts="${LAMBDA_LAUNCH_BURST_ATTEMPTS:-10}"
burst_interval="${LAMBDA_LAUNCH_BURST_INTERVAL:-5}"
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
  # Scarce types sell out in the seconds between the capacity check and the
  # launch call, but released capacity also flickers back within that window —
  # burst-retry the launch before falling back to slow polling.
  launched=0
  for (( attempt=1; attempt<=burst_attempts; attempt++ )); do
    if launch_instance; then launched=1; break; fi
    (( attempt < burst_attempts )) && { echo "  launch raced (attempt $attempt/$burst_attempts) — retrying in ${burst_interval}s"; sleep "$burst_interval"; }
  done
  (( launched )) && break
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
  else
    # A stale session from a previous run has the old instance's IP baked in,
    # so never reuse it — pick the first free altrux/altrux-N name.
    sess="altrux"; n=1
    while tmux has-session -t "$sess" 2>/dev/null; do sess="altrux-$n"; n=$((n + 1)); done
    [[ "$sess" != altrux ]] && echo "local tmux session 'altrux' already exists — using '$sess' for this run"
    # One local session, five views: watch = pull (top pane) + watchdog
    # (bottom pane); train/claude/work = live attaches to the remote tmux
    # sessions. The remote sessions don't exist until setup runs (and 'work',
    # where the experimenter runs prep/filter/probes, only when it first needs
    # one), so those windows poll until theirs appears, then attach.
    # ServerAliveInterval so a silently-dropped connection kills the ssh (and
    # its poll loop) instead of leaving the window waiting forever.
    rssh="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -i '$SSH_KEY_PATH' $SSH_USER@$ip"
    tmux new-session -d -s "$sess" -n watch "'$SCRIPT_DIR/lambda_pull.sh' --follow '$ip'; exec bash"
    tmux split-window -t "$sess:watch" "LAMBDA_INSTANCE_ID='$instance_id' LAMBDA_INSTANCE_IP='$ip' '$SCRIPT_DIR/lambda_watchdog.sh' --arm-after-training --pattern 'train.py|probe_recall.py|consolidation_null.py' --no-mem-state; exec bash"
    tmux new-window -t "$sess" -n train "$rssh -t 'until tmux has-session -t train 2>/dev/null; do echo \"waiting for remote train tmux...\"; sleep 5; done; exec tmux attach -t train'; exec bash"
    tmux new-window -t "$sess" -n claude "$rssh -t 'until tmux has-session -t experimenter 2>/dev/null; do echo \"waiting for remote experimenter tmux...\"; sleep 5; done; exec tmux attach -t experimenter'; exec bash"
    tmux new-window -t "$sess" -n work "$rssh -t 'until tmux has-session -t work 2>/dev/null; do echo \"waiting for remote work tmux...\"; sleep 5; done; exec tmux attach -t work'; exec bash"
    tmux select-window -t "$sess:watch"
    echo "Local tmux session '$sess' up — windows: watch (pull + watchdog panes), train (remote train tmux), claude (remote experimenter tmux), work (remote work tmux)."
    echo "Attach: tmux attach -t $sess"
  fi
fi

if [[ "$RUN_SETUP" == 0 ]]; then
  echo "Skipping setup (--no-setup). Finish with:"
  echo "  scp -i $SSH_KEY_PATH $SCRIPT_DIR/lambda_setup.sh $SSH_USER@$ip:"
  echo "  ssh -i $SSH_KEY_PATH $SSH_USER@$ip 'bash lambda_setup.sh'"
  exit 0
fi

if (( ${#all_ckpts[@]} + ${#data_local_files[@]} )); then
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "rm -rf ~/resume-staging && mkdir -p ~/resume-staging"
  RSYNC_SSH=(-e "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i '$SSH_KEY_PATH'")
fi
if (( ${#all_ckpts[@]} )); then
  echo "Uploading ${#all_ckpts[@]} checkpoint(s) (${#resume_ckpts_full[@]} with optimizer.pt, mem_state.pt excluded) to staging..."
  if (( ${#resume_ckpts[@]} )); then
    rsync -rtR --info=progress2 --exclude=mem_state.pt --exclude=optimizer.pt \
      "${RSYNC_SSH[@]}" "${resume_ckpts[@]}" "$SSH_USER@$ip:resume-staging/"
  fi
  # Full set second, so a step dir listed in both ends up with its optimizer.
  if (( ${#resume_ckpts_full[@]} )); then
    rsync -rtR --info=progress2 --exclude=mem_state.pt \
      "${RSYNC_SSH[@]}" "${resume_ckpts_full[@]}" "$SSH_USER@$ip:resume-staging/"
  fi
fi

if (( ${#data_local_files[@]} )); then
  echo "Uploading ${#data_local_files[@]} data artifact(s) to staging..."
  rsync -rtR --info=progress2 "${RSYNC_SSH[@]}" "${data_local_files[@]}" "$SSH_USER@$ip:resume-staging/"
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

# Stashed wheels (e.g. mamba-ssm harvested from a previous instance) save
# setup a multi-minute CUDA compile — uv falls back to building from source
# when no stashed wheel matches the box's torch/python combo.
wheel_dir="$SCRIPT_DIR/../.cache/wheels"
if compgen -G "$wheel_dir/*.whl" >/dev/null; then
  echo "Uploading stashed wheels ($(ls "$wheel_dir"))..."
  ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "mkdir -p ~/wheels"
  scp "${SSH_OPTS[@]}" "$wheel_dir"/*.whl "$SSH_USER@$ip:wheels/"
fi

echo "Uploading setup + config..."
scp "${SSH_OPTS[@]}" "$SCRIPT_DIR/lambda_setup.sh" "$SSH_USER@$ip:lambda_setup.sh"

# Forward config (incl. secrets) via a file rather than the command line, so
# tokens don't land in the instance's process list. Removed after setup reads it.
env_file="$(mktemp)"
trap 'rm -f "$env_file"' EXIT
GIT_USER_NAME="$(git config user.name 2>/dev/null || true)"
GIT_USER_EMAIL="$(git config user.email 2>/dev/null || true)"
# Mirror the local model/effort choices (settings.json stays local — see
# above), so the remote claude runs like this machine's claude is set to.
CLAUDE_MODEL="$(python3 -c 'import json,pathlib; print(json.loads((pathlib.Path.home()/".claude/settings.json").read_text()).get("model") or "")' 2>/dev/null || true)"
CLAUDE_EFFORT="$(python3 -c 'import json,pathlib; print(json.loads((pathlib.Path.home()/".claude/settings.json").read_text()).get("effortLevel") or "")' 2>/dev/null || true)"
for v in LAMBDA_REPO_URL LAMBDA_REMOTE_REPO LAMBDA_REPO_REF \
         CLAUDE_MODEL CLAUDE_EFFORT \
         GITHUB_TOKEN HF_TOKEN CLAUDE_CODE_OAUTH_TOKEN TORCH_BACKEND MAX_JOBS \
         GIT_USER_NAME GIT_USER_EMAIL; do
  [[ -n "${!v:-}" ]] && printf 'export %s=%q\n' "$v" "${!v}" >> "$env_file"
done
scp "${SSH_OPTS[@]}" "$env_file" "$SSH_USER@$ip:lambda_setup.env"

echo "Starting setup in detached tmux session '$SESSION'."
ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
  "command -v tmux >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq tmux; }; tmux new-session -d -s $SESSION 'set -a; . ~/lambda_setup.env 2>/dev/null; set +a; rm -f ~/lambda_setup.env; bash lambda_setup.sh; exec bash -l'"

echo
echo "Launched. Instance $instance_id is at $ip; setup is running in tmux '$SESSION'."
echo "Everything is viewable locally: tmux attach -t ${sess:-altrux} (windows: watch / train / claude)."
echo "Direct ssh fallback: ssh -i $SSH_KEY_PATH $SSH_USER@$ip -t tmux attach -t <$SESSION|experimenter>"
