#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../experimenter_agent.sh
source "$ROOT/scripts/experimenter_agent.sh"

[[ "$(experimenter_prompt claude brief)" == "/altrux-experimenter brief" ]]
[[ "$(experimenter_prompt codex brief)" == '$altrux-experimenter brief' ]]

experimenter_command claude "prompt with spaces"
[[ "${EXPERIMENTER_COMMAND[*]}" == "claude --dangerously-skip-permissions prompt with spaces" ]]

experimenter_command codex "prompt with spaces"
[[ "${EXPERIMENTER_COMMAND[*]}" == "codex --dangerously-bypass-approvals-and-sandbox prompt with spaces" ]]

experimenter_agent_validate claude
experimenter_agent_validate codex
if experimenter_agent_validate unknown 2>/dev/null; then
  echo "unknown provider accepted" >&2
  exit 1
fi

launch="$ROOT/scripts/lambda_launch.sh"
grep -Fq 'EXPERIMENTER_AGENT=$EXPERIMENTER_AGENT bash lambda_setup.sh' "$launch"
if grep -Fq 'for f in AGENTS.md config.toml auth.json' "$launch"; then
  echo "launch must not upload full Codex config" >&2
  exit 1
fi
grep -Fq 'for f in auth.json' "$launch"

echo "experimenter agent tests passed"
