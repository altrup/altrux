#!/usr/bin/env bash
# Configures a freshly-launched Lambda Cloud GPU instance for a training run:
# clone/pull the repo, `make sync` (+ verify CUDA torch), prepare data, and
# install the Claude Code CLI so a `/experimenter` session can take over.
#
# Runs ON the instance (not the local machine — it holds no Lambda API key and
# needs none). Either run it by hand after ssh-ing in, or let lambda_launch.sh
# scp it up and run it automatically. Idempotent: safe to re-run after a
# partial failure (clone-or-pull, sync re-runs cleanly, data prep skips if its
# output already exists).
#
# Config (all optional, via environment):
#   LAMBDA_REPO_URL        git URL to clone       (default: repo's origin)
#   LAMBDA_REMOTE_REPO     dir name under $HOME   (default: altrux)
#   LAMBDA_REPO_REF        branch/tag/commit      (default: repo default)
#   LAMBDA_SETUP_DATA_CMD  data-prep make target  (default: make data-memory)
#   LAMBDA_SETUP_DATA_MARKER  skip data prep if this file exists, relative to
#                             the repo root (default: sft/data/train_memory.pt);
#                             set empty to always run.
#   GITHUB_TOKEN           if set, configures git to clone+push over HTTPS
#   HF_TOKEN               if set, exported for gated/large HF downloads and
#                          persisted to ~/.bashrc for later sessions
#   TORCH_BACKEND          passed to `make sync` (e.g. cu128 for GH200)
#   MAX_JOBS               parallel compile jobs for `make sync`
set -euo pipefail

REPO_URL="${LAMBDA_REPO_URL:-https://github.com/altrup/altrux.git}"
REPO_DIR="$HOME/${LAMBDA_REMOTE_REPO:-altrux}"
REPO_REF="${LAMBDA_REPO_REF:-}"
MODEL_NAME="${LAMBDA_MODEL_NAME:-mamba2_2_7b_memory}"
DATA_CMD="${LAMBDA_SETUP_DATA_CMD:-make data data-memory prepare-chains}"
DATA_MARKER="${LAMBDA_SETUP_DATA_MARKER-sft/data/train_chains.pt}"

step() { printf '\n=== %s ===\n' "$1"; }

step "Bootstrap: uv + credentials"
command -v git >/dev/null || { echo "error: git not found on instance" >&2; exit 1; }
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
[[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  git config --global \
    url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  grep -q '^export HF_TOKEN=' "$HOME/.bashrc" 2>/dev/null \
    || echo "export HF_TOKEN=${HF_TOKEN}" >> "$HOME/.bashrc"
fi
if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
  export CLAUDE_CODE_OAUTH_TOKEN
  grep -q '^export CLAUDE_CODE_OAUTH_TOKEN=' "$HOME/.bashrc" 2>/dev/null \
    || echo "export CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}" >> "$HOME/.bashrc"
fi

step "Repo: $REPO_DIR"
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch --prune
  [[ -n "$REPO_REF" ]] && git -C "$REPO_DIR" checkout "$REPO_REF"
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
  [[ -n "$REPO_REF" ]] && git -C "$REPO_DIR" checkout "$REPO_REF"
fi

step "Stage resume checkpoint"
if [[ -d "$HOME/resume-staging" && -n "$(ls -A "$HOME/resume-staging" 2>/dev/null)" ]]; then
  dest="$REPO_DIR/models/$MODEL_NAME/checkpoints/${LAMBDA_RESUME_EPOCH:-epoch-1}"
  mkdir -p "$dest"
  mv "$HOME"/resume-staging/* "$dest"/
  rmdir "$HOME/resume-staging" 2>/dev/null || true
  echo "placed $(ls "$dest") in $dest — training will resume from it"
else
  echo "no checkpoint staged — training will start fresh"
fi

step "Configure sft/.env (MODEL_NAME=$MODEL_NAME)"
[[ -f "$REPO_DIR/sft/.env" ]] || cp "$REPO_DIR/sft/.env.example" "$REPO_DIR/sft/.env"
sed -i "s/^MODEL_NAME=.*/MODEL_NAME=$MODEL_NAME/" "$REPO_DIR/sft/.env"

step "make sync (torch + mamba-ssm — several minutes)"
make -C "$REPO_DIR/sft" sync

step "Verify CUDA torch"
if ! uv run --project "$REPO_DIR/sft" --no-sync python -c \
  "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok); sys.exit(0 if ok else 1)"; then
  echo "error: torch reports no CUDA GPU — sync may have installed a CPU/ROCm build; investigate before training" >&2
  exit 1
fi

step "Data prep: $DATA_CMD"
if [[ -n "$DATA_MARKER" && -f "$REPO_DIR/$DATA_MARKER" ]]; then
  echo "skipping — $DATA_MARKER already exists (set LAMBDA_SETUP_DATA_MARKER= to force)"
else
  ( cd "$REPO_DIR/sft" && eval "$DATA_CMD" )
fi

step "Verify git push auth"
if ! git -C "$REPO_DIR" push --dry-run origin HEAD; then
  echo "warning: 'git push --dry-run' failed — the instance can't push fixes." >&2
  echo "         Set up a deploy key / credential before relying on /experimenter." >&2
fi

step "Install Claude Code CLI"
if command -v claude >/dev/null 2>&1; then
  echo "claude already installed: $(command -v claude)"
elif command -v npm >/dev/null 2>&1; then
  npm install -g @anthropic-ai/claude-code
else
  curl -fsSL https://claude.ai/install.sh | bash
fi

EXP_PROMPT="/experimenter You were started automatically by the setup script on a freshly provisioned instance. Your human teammates set this up and may be AFK, so operate autonomously within the brief and the watchdog cost controls: read the prior notes, then start training in the train session and monitor it."

step "Start the 'experimenter' tmux session"
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found — run the experimenter however you like"
  auto=0
elif tmux has-session -t experimenter 2>/dev/null; then
  echo "session 'experimenter' already exists — leaving it as-is"
  auto=0
elif [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]] && command -v claude >/dev/null 2>&1; then
  tmux new-session -d -s experimenter -c "$REPO_DIR" \
    "claude --dangerously-skip-permissions '$EXP_PROMPT'"
  echo "auto-started claude /experimenter (CLAUDE_CODE_OAUTH_TOKEN present)"
  auto=1
else
  tmux new-session -d -s experimenter -c "$REPO_DIR"
  echo "session 'experimenter' ready (no token — start claude by hand)"
  auto=0
fi

cat <<EOF

=== Setup complete ===
'train' tmux session: training runs here (leave it at this shell).
'experimenter' tmux session: the monitoring Claude session.
EOF
if [[ "$auto" == 1 ]]; then
  echo "  It is already running autonomously — watch it with: tmux attach -t experimenter"
else
  echo "  Start it: tmux attach -t experimenter, run 'claude' (paste a"
  echo "  'claude setup-token' value to auth), then /experimenter."
fi
