#!/usr/bin/env bash
# Configures a freshly-launched Lambda Cloud GPU instance for a training run:
# turn the rsync-uploaded repo into a git checkout of this run's `box/<UTC>`
# branch (the experimenter commits and pushes code and notes there), `make
# sync` (+ verify CUDA torch), and install the selected agent CLI so an
# experimenter session can take over. Data prep is deliberately NOT done
# here — which data (and with what flags) is an experimental decision the
# experimenter makes from the notes.
#
# Runs ON the instance (not the local machine — it holds no Lambda API key and
# needs none). Either run it by hand after ssh-ing in, or let lambda_launch.sh
# scp it up and run it automatically. Idempotent: safe to re-run after a
# partial failure (sync re-runs cleanly).
#
# Config (all optional, via environment):
#   LAMBDA_REMOTE_REPO     dir name under $HOME   (default: altrux)
#   HF_TOKEN               if set, exported for gated/large HF downloads and
#                          persisted to ~/.bashrc for later sessions
#   TORCH_BACKEND          passed to `make sync` (e.g. cu128 for GH200)
#   MAX_JOBS               parallel compile jobs for `make sync`
#   EXPERIMENTER_AGENT     claude or codex (default: claude)
#   BOX_BRANCH             this run's branch, pushed to origin by launch
#   GITHUB_TOKEN           pushes to that branch; stored in the repo's git
#                          config only (never global, never in the shell)
#   LAMBDA_REPO_URL        origin (default: https://github.com/altrup/altrux.git)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=experimenter_agent.sh
source "$SCRIPT_DIR/experimenter_agent.sh"
EXPERIMENTER_AGENT="${EXPERIMENTER_AGENT:-claude}"
experimenter_agent_validate "$EXPERIMENTER_AGENT"

REPO_DIR="$HOME/${LAMBDA_REMOTE_REPO:-altrux}"

step() { printf '\n=== %s ===\n' "$1"; }

step "Bootstrap: uv + credentials"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
[[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env"

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

step "Repo: $REPO_DIR on branch $BOX_BRANCH"
[[ -f "$REPO_DIR/.upload-rev" ]] || { echo "error: $REPO_DIR was not uploaded by lambda_launch.sh (no .upload-rev)" >&2; exit 1; }
[[ -n "${BOX_BRANCH:-}" && -n "${GITHUB_TOKEN:-}" ]] || { echo "error: BOX_BRANCH and GITHUB_TOKEN are required (lambda_launch.sh forwards them)" >&2; exit 1; }
echo "local revision: $(cat "$REPO_DIR/.upload-rev")"
command -v git >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git
REPO_URL="${LAMBDA_REPO_URL:-https://github.com/altrup/altrux.git}"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  # The rsynced tree is the working tree; the branch launch pushed is its
  # index, so `git status` shows exactly the uncommitted edits that came up.
  git -C "$REPO_DIR" init -q
  git -C "$REPO_DIR" remote add origin "$REPO_URL"
  git -C "$REPO_DIR" config "url.https://x-access-token:${GITHUB_TOKEN}@github.com/.insteadOf" "https://github.com/"
  git -C "$REPO_DIR" config user.name "altrux box"
  git -C "$REPO_DIR" config user.email "box@altrux.invalid"
  git -C "$REPO_DIR" config core.hooksPath scripts/git-hooks
  git -C "$REPO_DIR" fetch -q --depth=1 origin "+refs/heads/$BOX_BRANCH:refs/remotes/origin/$BOX_BRANCH"
  git -C "$REPO_DIR" reset -q "origin/$BOX_BRANCH"
  git -C "$REPO_DIR" branch -q -m "$BOX_BRANCH"
  git -C "$REPO_DIR" branch -q -u "origin/$BOX_BRANCH"
  rm -f "$REPO_DIR/.upload-rev"
  if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    git -C "$REPO_DIR" add -A
    git -C "$REPO_DIR" -c core.hooksPath=/dev/null commit -q -m "box: uncommitted local changes as uploaded"
    git -C "$REPO_DIR" push -q origin "$BOX_BRANCH"
    echo "committed and pushed the uploaded working-tree changes"
  fi
fi
echo "branch: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) at $(git -C "$REPO_DIR" rev-parse --short HEAD)"

step "Stage uploaded checkpoints + data artifacts"
# lambda_launch.sh uploads both path-preserving (repo-relative), so staging
# already mirrors models/<model>/checkpoints/... and sft/data/ — merge it
# verbatim. Staged data artifacts are finished datasets: regenerate only what
# isn't listed here.
if [[ -d "$HOME/resume-staging" && -n "$(ls -A "$HOME/resume-staging" 2>/dev/null)" ]]; then
  find "$HOME/resume-staging" -mindepth 1 -maxdepth 4 -type d -name 'step-*' -printf 'placing checkpoint %P\n'
  [[ -d "$HOME/resume-staging/sft/data" ]] &&
    find "$HOME/resume-staging/sft/data" -maxdepth 1 -type f -printf 'placing data artifact sft/data/%P (already generated — do not regenerate)\n'
  cp -a "$HOME/resume-staging/." "$REPO_DIR/"
  rm -rf "$HOME/resume-staging"
else
  echo "nothing staged — training starts fresh and all data must be generated here"
fi

step "Configure sft/.env"
[[ -f "$REPO_DIR/sft/.env" ]] || cp "$REPO_DIR/sft/.env.example" "$REPO_DIR/sft/.env"
# Blank MODEL_NAME so a run without an explicit choice fails at import
# instead of silently training the example default. The experimenter passes
# MODEL_NAME=<arm> inline on every training/probe/filter command (inline wins
# over .env), per the DISCUSSION notes.
sed -i "s/^MODEL_NAME=.*/MODEL_NAME=/" "$REPO_DIR/sft/.env"
echo "MODEL_NAME left blank — pass MODEL_NAME=<arm> inline on each command"

step "make sync (torch + mamba-ssm — several minutes)"
# Wheels uploaded through LAMBDA_CACHE_ARTIFACTS (harvested from a previous instance)
# spare the mamba-ssm CUDA compile; a stale/mismatched wheel is simply not
# selected and uv builds from source as usual.
if ls "$REPO_DIR"/.cache/wheels/*.whl >/dev/null 2>&1; then
  echo "using stashed wheels from $REPO_DIR/.cache/wheels: $(ls "$REPO_DIR"/.cache/wheels)"
  export UV_FIND_LINKS="$REPO_DIR/.cache/wheels"
fi
make -C "$REPO_DIR/sft" sync

step "Verify CUDA torch"
if ! uv run --project "$REPO_DIR/sft" --no-sync python -c \
  "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok); sys.exit(0 if ok else 1)"; then
  echo "error: torch reports no CUDA GPU — sync may have installed a CPU/ROCm build; investigate before training" >&2
  exit 1
fi

step "Install $EXPERIMENTER_AGENT CLI"
if [[ "$EXPERIMENTER_AGENT" == claude ]]; then
  if command -v claude >/dev/null 2>&1; then
    echo "claude already installed: $(command -v claude)"
  elif command -v npm >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code
  else
    curl -fsSL https://claude.ai/install.sh | bash
  fi
elif command -v codex >/dev/null 2>&1; then
  echo "codex already installed: $(command -v codex)"
else
  curl -fsSL https://chatgpt.com/codex/install.sh | sh
fi
[[ -f "$HOME/.local/bin/env" ]] && . "$HOME/.local/bin/env"

# Pre-answer claude's interactive first-run prompts (onboarding/theme, the
# per-folder trust dialog, the --dangerously-skip-permissions confirm) — the
# auto-started experimenter session would otherwise sit blocked on them until
# a human attaches. Also wire in the status line if lambda_launch.sh uploaded
# one (it ships statusline.sh but deliberately not settings.json).
if [[ "$EXPERIMENTER_AGENT" == claude ]]; then
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
fi

EXP_BRIEF="You were started automatically by the setup script on a freshly provisioned instance. Your teammates may be AFK, so operate autonomously within the brief and watchdog cost controls. Read notes/README.md and follow its current reading path. Decide what this session should do first; that may be evaluations or probes instead of training. Execute long work in the registered tmux session."
EXP_PROMPT="$(experimenter_prompt "$EXPERIMENTER_AGENT" "$EXP_BRIEF")"
experimenter_command "$EXPERIMENTER_AGENT" "$EXP_PROMPT"
printf -v EXPERIMENTER_SHELL_COMMAND '%q ' "${EXPERIMENTER_COMMAND[@]}"

step "Start the 'experimenter' tmux session"
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found — run the experimenter however you like"
  auto=0
elif tmux has-session -t experimenter 2>/dev/null; then
  echo "session 'experimenter' already exists — leaving it as-is"
  auto=0
elif [[ "$EXPERIMENTER_AGENT" == claude && -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]] && command -v claude >/dev/null 2>&1; then
  # tmux panes are children of the tmux server, not of this script — they
  # inherit neither the exported token nor ~/.local/bin on PATH (and Ubuntu's
  # .bashrc exits before the appended exports in non-interactive shells), so
  # pass both into the session explicitly.
  tmux new-session -d -s experimenter -c "$REPO_DIR" \
    -e CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" -e PATH="$PATH" \
    "$EXPERIMENTER_SHELL_COMMAND"
  echo "auto-started claude /altrux-experimenter (CLAUDE_CODE_OAUTH_TOKEN present)"
  auto=1
elif [[ "$EXPERIMENTER_AGENT" == codex ]] && command -v codex >/dev/null 2>&1 \
    && codex login status >/dev/null 2>&1; then
  tmux new-session -d -s experimenter -c "$REPO_DIR" -e PATH="$PATH" \
    "$EXPERIMENTER_SHELL_COMMAND"
  echo "auto-started codex \$altrux-experimenter (cached login present)"
  auto=1
else
  tmux new-session -d -s experimenter -c "$REPO_DIR"
  echo "session 'experimenter' ready (authenticate and start $EXPERIMENTER_AGENT by hand)"
  auto=0
fi

cat <<EOF

=== Setup complete ===
'train' tmux session: training runs here (leave it at this shell).
'experimenter' tmux session: the monitoring $EXPERIMENTER_AGENT session.
EOF
if [[ "$auto" == 1 ]]; then
  echo "  It is already running autonomously — watch it with: tmux attach -t experimenter"
else
  echo "  Attach with: tmux attach -t experimenter"
  if [[ "$EXPERIMENTER_AGENT" == claude ]]; then
    echo "  Authenticate Claude, then invoke /altrux-experimenter."
  else
    echo "  Run 'codex login --device-auth', then invoke \$altrux-experimenter."
  fi
fi
