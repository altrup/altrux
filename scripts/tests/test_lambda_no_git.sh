#!/usr/bin/env bash
# The box never touches git: no token, no clone, no push; the repo goes up by rsync.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launch="$ROOT/scripts/lambda_launch.sh"; setup="$ROOT/scripts/lambda_setup.sh"
skill="$ROOT/.agents/skills/altrux-experimenter/SKILL.md"

! grep -qE 'GITHUB_TOKEN|GIT_USER_NAME|LAMBDA_REPO_(URL|REF)' "$launch" "$setup" "$ROOT/scripts/.env.example"
! grep -qE 'git (clone|pull|push|fetch|config)' "$setup"
grep -q 'git -C "$ROOT" ls-files -z --cached --others --exclude-standard' "$launch"
grep -q 'cp -a "$REPO_DIR" "$HOME/pristine"' "$setup"
grep -q 'ALTRUX_PROTECT_OFF' "$setup"
! grep -qiE 'commit and push|rescue/' "$skill"
grep -q 'scripts/box_patch.sh' "$skill"
bash -n "$launch" "$setup" "$ROOT/scripts/box_patch.sh"

echo "lambda no-git tests passed"
