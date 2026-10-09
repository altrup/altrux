#!/usr/bin/env bash
# The box works on its own box/<UTC> branch: launch pushes it, setup checks it out, nothing patches.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launch="$ROOT/scripts/lambda_launch.sh"; setup="$ROOT/scripts/lambda_setup.sh"
skill="$ROOT/.agents/skills/altrux-experimenter/SKILL.md"

grep -q 'require GITHUB_TOKEN' "$launch"
grep -q 'BOX_BRANCH="box/' "$launch"
grep -q 'git -C "$ROOT" push -q origin "HEAD:refs/heads/$BOX_BRANCH"' "$launch"
grep -q 'git -C "$ROOT" ls-files -z --cached --others --exclude-standard' "$launch"
grep -qE 'forward_vars=\(.*BOX_BRANCH GITHUB_TOKEN' "$launch"
grep -q 'git -C "$REPO_DIR" config core.hooksPath scripts/git-hooks' "$setup"
grep -q 'git -C "$REPO_DIR" branch -q -m "$BOX_BRANCH"' "$setup"
! grep -q 'git config --global' "$setup"
! grep -qE 'ALTRUX_PROTECT_OFF|pristine' "$setup"
grep -q 'box/<UTC>' "$skill"
! grep -qiE 'box_patch|pristine|rescue/' "$skill" "$launch" "$setup" "$ROOT/scripts/README.md" "$ROOT/scripts/.env.example"
bash -n "$launch" "$setup"

echo "lambda box-branch tests passed"
