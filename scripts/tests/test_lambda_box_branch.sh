#!/usr/bin/env bash
# The box works on its own box/<UTC> branch: launch pushes it, setup checks it out, nothing patches.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launch="$ROOT/scripts/lambda_launch.sh"; setup="$ROOT/scripts/lambda_setup.sh"
skill="$ROOT/.agents/skills/altrux-experimenter/SKILL.md"
# `! grep` never trips set -e, so negative checks go through this.
absent() { if grep -qiE "$1" "${@:2}"; then echo "unexpected '$1' in ${*:2}" >&2; exit 1; fi; }

grep -q 'require GITHUB_TOKEN' "$launch"
grep -q 'BOX_BRANCH="box/' "$launch"
absent 'git -C "\$ROOT" push' "$launch"
absent "upload-rev|git -C .$ROOT. push" "$launch" "$setup"
grep -q 'branch -r --contains HEAD' "$launch"
grep -q 'ls-files --others --exclude-standard' "$launch"
grep -qE 'forward_vars=\(.*BOX_BRANCH BOX_BASE GITHUB_TOKEN' "$launch"
grep -q 'git -C "$REPO_DIR" config core.hooksPath scripts/git-hooks' "$setup"
grep -q 'git -C "$REPO_DIR" fetch -q --depth=1 origin "$BOX_BASE"' "$setup"
grep -q 'git -C "$REPO_DIR" checkout -q -b "$BOX_BRANCH" FETCH_HEAD' "$setup"
grep -q 'git -C "$REPO_DIR" push -q -u origin "$BOX_BRANCH"' "$setup"
absent 'git config --global' "$setup"
absent 'ALTRUX_PROTECT_OFF|pristine' "$setup"
grep -q 'box/<UTC>' "$skill"
absent 'box_patch|pristine|rescue/' "$skill" "$launch" "$setup" "$ROOT/scripts/README.md" "$ROOT/scripts/.env.example"
bash -n "$launch" "$setup"

echo "lambda box-branch tests passed"
