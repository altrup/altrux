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

## No spaCy in this venv

`prepare_cram.py` needs NER and the next-run plan called for spaCy, but this venv is Python 3.14 and spaCy publishes no cp314 wheels — `uv pip install spacy` resolves, then source-builds thinc/blis/murmurhash/preshed/cymem/srsly, on top of the standing risk that any `uv` install re-resolves torch off the ROCm build (root `CLAUDE.md`). It uses `transformers`' CoNLL-03 token classifier instead (already a dependency, PER/ORG/LOC is all a same-type entity swap needs). Don't re-attempt spaCy without checking for cp314 wheels first.

## Data-shape gotcha: mid-conversation sleeps need episodes that don't exist

`prepare_chains.py --mid-sleep-rate` only fires on episodes ≥ `--mid-sleep-min-len` (4096 default; 1536 used in practice) that have a turn boundary in their middle third. LongAlign/babilong episodes are single-QA (turn boundaries only at position 0 and the answer start) so they never qualify; ultrachat only qualifies when tokenized uncapped (`prepare_data.py --max-len`, default 32768 — the old 1024 cap made the set empty and mid-sleep a silent no-op in every dataset before 2026-07-21). Always check the regen log's mid-conversation counter before assuming the signal exists. `--split-episode-rate` (with `--split-min-part`) creates natural-continuation-across-sleep signal from multi-turn corpora; single-QA episodes only split (`--split-qa-rate`) when they carry a `question_offsets` entry from `prepare_data.py` (babilong yes, LongAlign never — fail closed); `--sentence-sleep-rate` reaches inside single-QA document turns via sentence boundaries (built 2026-07-21, deliberately unused until the 396-reproduction question is settled — see `notes/DISCUSSION-20260721-peak-reproducibility.md`).
