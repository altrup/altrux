# Claude Guidelines — sft

## Always update the README

Whenever you change user-facing behaviour — new CLI flags, changed defaults, a removed feature — update `sft/README.md` in the same change. Keep the usage examples and key flags accurate; point users to `--help` for the full option list.

## Cache policy

Follow the root `CLAUDE.md` cache policy: all caches go in `.cache/` (gitignored). Every `Makefile` target that downloads weights or packages must set `HF_HOME=.cache/huggingface` and `UV_CACHE_DIR=.cache/uv`.
