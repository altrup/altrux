# Claude Guidelines — sft

## Always update the README

Whenever you change user-facing behaviour — new CLI flags, changed defaults, a removed feature — update `sft/README.md` in the same change. Keep the usage examples and key flags accurate; point users to `--help` for the full option list.

## Cache policy

HuggingFace cache is shared at the repo root (`../.cache/huggingface`), not inside this folder. The Makefile sets `HF_HOME=$(CURDIR)/../.cache/huggingface` on every relevant target. See the root `CLAUDE.md` for the full policy.

uv uses its default system cache — no override needed.
