#!/usr/bin/env bash

experimenter_agent_validate() {
  case "$1" in
    claude|codex) ;;
    *) echo "error: EXPERIMENTER_AGENT must be claude or codex (got '$1')" >&2; return 2 ;;
  esac
}

experimenter_prompt() {
  case "$1" in
    claude) printf '/altrux-experimenter %s' "$2" ;;
    codex) printf '$altrux-experimenter %s' "$2" ;;
  esac
}

experimenter_command() {
  case "$1" in
    claude) EXPERIMENTER_COMMAND=(claude --dangerously-skip-permissions "$2") ;;
    codex) EXPERIMENTER_COMMAND=(codex --dangerously-bypass-approvals-and-sandbox "$2") ;;
  esac
}
