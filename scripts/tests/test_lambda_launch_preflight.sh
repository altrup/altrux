#!/usr/bin/env bash
# Launch verifies CLAUDE_CODE_OAUTH_TOKEN before the first Lambda API call.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
# A copy, so the real scripts/.env is never read; ROOT becomes $work/repo.
mkdir -p "$work/repo/scripts" "$work/bin"
cp "$ROOT"/scripts/{lambda_launch.sh,lambda_data_artifacts.sh,lambda_cache_artifacts.sh,experimenter_agent.sh} "$work/repo/scripts/"
touch "$work/key"

cat > "$work/bin/git" <<'STUB'
#!/usr/bin/env bash
case "$*" in
  *"branch -r"*) echo origin/main ;;
  *rev-parse*) echo 0123456789abcdef0123456789abcdef01234567 ;;
esac
STUB
cat > "$work/bin/curl" <<STUB
#!/usr/bin/env bash
echo "\$*" >> "$work/curl.log"
exit 1
STUB
cat > "$work/bin/claude" <<'STUB'
#!/usr/bin/env bash
[[ -z "${ANTHROPIC_API_KEY:-}" && "$CLAUDE_CODE_OAUTH_TOKEN" == good ]] || { echo "401 invalid" >&2; exit 1; }
echo ok
STUB
chmod +x "$work/bin/"*

launch() {
  rm -f "$work/curl.log"
  PATH="$work/bin:$PATH" EXPERIMENTER_AGENT=claude CLAUDE_CODE_OAUTH_TOKEN="$1" ANTHROPIC_API_KEY=shadow \
    LAMBDA_API_KEY=k LAMBDA_INSTANCE_TYPE=t LAMBDA_SSH_KEY_NAME=n GITHUB_TOKEN=g LAMBDA_SSH_KEY_PATH="$work/key" \
    LAMBDA_CAPACITY_POLL_INTERVAL=1 LAMBDA_CAPACITY_MAX_WAIT=1 \
    timeout 30 bash "$work/repo/scripts/lambda_launch.sh" < /dev/null > "$work/out" 2>&1
}

if launch bad; then fail "launch proceeded with a rejected token: $(cat "$work/out")"; fi
[[ ! -e "$work/curl.log" ]] || fail "the Lambda API was called before the token check failed"

if launch ""; then fail "launch proceeded with no token"; fi
[[ ! -e "$work/curl.log" ]] || fail "the Lambda API was called with no token"

# Gives up at the capacity poll: the stubbed API never answers.
launch good || true
grep -q instance-types "$work/curl.log" 2>/dev/null || fail "a working token did not reach the rent step: $(cat "$work/out")"

echo "lambda launch preflight tests passed"
