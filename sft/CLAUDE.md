# Claude Guidelines — sft

## Always update the README

Whenever you change user-facing behaviour — new CLI flags, changed defaults, a removed feature — update `sft/README.md` in the same change. Keep the usage examples and key flags accurate; point users to `--help` for the full option list.

Whenever you add, rename, or remove an environment variable, also update `sft/.env.example`.

## Cache policy

HuggingFace cache is shared at the repo root (`../.cache/huggingface`), not inside this folder. The Makefile sets `HF_HOME=$(CURDIR)/../.cache/huggingface` on every relevant target. See the root `CLAUDE.md` for the full policy.

uv uses its default system cache — no override needed.

## Model configuration

The model is selected via `MODEL_NAME` in `.env`. This controls which package in `models/` is imported. The scripts add the repo root to `sys.path` automatically; the Makefile also sets `PYTHONPATH`. See the root `CLAUDE.md` for the full model interface contract.

## Checkpoint format

`state.pt` (written by `train.py`'s `save_checkpoint`) carries a `dataset_fingerprint` (resolved `--data` path + example count) alongside `slot_states`/`next_ptr`, so `main()`'s resume path can tell whether `--data` still points at the dataset those indices were recorded against and reset them instead of silently reindexing into an unrelated dataset. If you touch `state.pt`'s schema again, keep this field — `slot_states`/`next_ptr` are meaningless without it.
