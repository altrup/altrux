#!/usr/bin/env bash
# Configures a freshly-launched Lambda Cloud GPU instance for a training run:
# clone/pull the repo, `make sync` (+ verify CUDA torch), and install the
# Claude Code CLI so a `/experimenter` session can take over. Data prep is
# deliberately NOT done here — which data (and with what flags) is an
# experimental decision the experimenter makes from the notes.
#
# Runs ON the instance (not the local machine — it holds no Lambda API key and
# needs none). Either run it by hand after ssh-ing in, or let lambda_launch.sh
# scp it up and run it automatically. Idempotent: safe to re-run after a
# partial failure (clone-or-pull, sync re-runs cleanly).
#
# Config (all optional, via environment):
#   LAMBDA_REPO_URL        git URL to clone       (default: repo's origin)
#   LAMBDA_REMOTE_REPO     dir name under $HOME   (default: altrux)
#   LAMBDA_REPO_REF        branch/tag/commit      (default: repo default)
#   GITHUB_TOKEN           if set, configures git to clone+push over HTTPS
#   HF_TOKEN               if set, exported for gated/large HF downloads and
#                          persisted to ~/.bashrc for later sessions
#   TORCH_BACKEND          passed to `make sync` (e.g. cu128 for GH200)
#   MAX_JOBS               parallel compile jobs for `make sync`
set -euo pipefail

REPO_URL="${LAMBDA_REPO_URL:-https://github.com/altrup/altrux.git}"
REPO_DIR="$HOME/${LAMBDA_REMOTE_REPO:-altrux}"
REPO_REF="${LAMBDA_REPO_REF:-}"

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
# Forwarded by lambda_launch.sh from the local machine's git config, so
# commits made on the instance carry the same identity.
[[ -n "${GIT_USER_NAME:-}" ]] && git config --global user.name "$GIT_USER_NAME"
[[ -n "${GIT_USER_EMAIL:-}" ]] && git config --global user.email "$GIT_USER_EMAIL"
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

step "Stage resume checkpoints"
# lambda_launch.sh uploads checkpoints path-preserving (repo-relative), so
# staging already mirrors models/<model>/checkpoints/... — merge it verbatim.
if [[ -d "$HOME/resume-staging" && -n "$(ls -A "$HOME/resume-staging" 2>/dev/null)" ]]; then
  find "$HOME/resume-staging" -mindepth 1 -maxdepth 4 -type d -name 'step-*' -printf 'placing %P\n'
  cp -a "$HOME/resume-staging/." "$REPO_DIR/"
  rm -rf "$HOME/resume-staging"
else
  echo "no checkpoint staged — training will start fresh"
fi

step "Configure sft/.env"
[[ -f "$REPO_DIR/sft/.env" ]] || cp "$REPO_DIR/sft/.env.example" "$REPO_DIR/sft/.env"
# Blank MODEL_NAME so a run without an explicit choice fails at import
# instead of silently training the example default. The experimenter sets it
# per training leg from the DISCUSSION notes.
sed -i "s/^MODEL_NAME=.*/MODEL_NAME=/" "$REPO_DIR/sft/.env"
echo "MODEL_NAME left blank — set it in sft/.env before each training leg"

step "make sync (torch + mamba-ssm — several minutes)"
# Wheels uploaded by lambda_launch.sh (harvested from a previous instance)
# spare the mamba-ssm CUDA compile; a stale/mismatched wheel is simply not
# selected and uv builds from source as usual.
if ls "$HOME"/wheels/*.whl >/dev/null 2>&1; then
  echo "using stashed wheels from ~/wheels: $(ls "$HOME"/wheels)"
  export UV_FIND_LINKS="$HOME/wheels"
fi
make -C "$REPO_DIR/sft" sync

step "Verify CUDA torch"
if ! uv run --project "$REPO_DIR/sft" --no-sync python -c \
  "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok); sys.exit(0 if ok else 1)"; then
  echo "error: torch reports no CUDA GPU — sync may have installed a CPU/ROCm build; investigate before training" >&2
  exit 1
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

# Pre-answer claude's interactive first-run prompts (onboarding/theme, the
# per-folder trust dialog, the --dangerously-skip-permissions confirm) — the
# auto-started experimenter session would otherwise sit blocked on them until
# a human attaches. Also wire in the status line if lambda_launch.sh uploaded
# one (it ships statusline.sh but deliberately not settings.json).
[[ -f "$HOME/.claude/statusline.sh" ]] && ! command -v jq >/dev/null \
  && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq jq
REPO_DIR="$REPO_DIR" python3 - <<'EOF'
import json, os, pathlib
home = pathlib.Path.home()

p = home / ".claude/settings.json"
s = json.loads(p.read_text()) if p.exists() else {}
s.setdefault("theme", "dark")
if os.environ.get("CLAUDE_MODEL"):
    s["model"] = os.environ["CLAUDE_MODEL"]
if os.environ.get("CLAUDE_EFFORT"):
    s["effortLevel"] = os.environ["CLAUDE_EFFORT"]
s["skipDangerousModePermissionPrompt"] = True
if (home / ".claude/statusline.sh").exists():
    s["statusLine"] = {"type": "command", "command": "bash ~/.claude/statusline.sh", "refreshInterval": 1}
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(s, indent=2) + "\n")

cj = home / ".claude.json"
d = json.loads(cj.read_text()) if cj.exists() else {}
d["hasCompletedOnboarding"] = True
d.setdefault("projects", {}).setdefault(os.environ["REPO_DIR"], {})["hasTrustDialogAccepted"] = True
cj.write_text(json.dumps(d, indent=2) + "\n")
EOF
echo "first-run prompts pre-answered (onboarding, trust, skip-permissions confirm)"

EXP_PROMPT="/experimenter You were started automatically by the setup script on a freshly provisioned instance. Your teammates set this up and may be AFK, so operate autonomously within the brief and the watchdog cost controls: read the prior notes and any DISCUSSION notes, decide what this session should do first (that may be evals/probes rather than training — the DISCUSSION notes carry the current plan), and execute it in the train session."

step "Start the 'experimenter' tmux session"
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found — run the experimenter however you like"
  auto=0
elif tmux has-session -t experimenter 2>/dev/null; then
  echo "session 'experimenter' already exists — leaving it as-is"
  auto=0
elif [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]] && command -v claude >/dev/null 2>&1; then
  # tmux panes are children of the tmux server, not of this script — they
  # inherit neither the exported token nor ~/.local/bin on PATH (and Ubuntu's
  # .bashrc exits before the appended exports in non-interactive shells), so
  # pass both into the session explicitly.
  tmux new-session -d -s experimenter -c "$REPO_DIR" \
    -e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" -e PATH="$PATH" \
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
