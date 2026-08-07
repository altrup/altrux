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

`--sleep-chain-rate` (0.1) gates *all* of the below: an episode's rates only apply if its chain drew into the sleeping fraction, so every sleep/split/fact count in a regen log is ~10% of what the per-episode rates alone imply. Then `prepare_chains.py --mid-sleep-rate` only fires on episodes ≥ `--mid-sleep-min-len` (4096 default; 1536 used in practice) that have a turn boundary in their middle third. LongAlign/babilong episodes are single-QA (turn boundaries only at position 0 and the answer start) so they never qualify; ultrachat only qualifies when tokenized uncapped (`prepare_data.py --max-len`, default 32768 — the old 1024 cap made the set empty and mid-sleep a silent no-op in every dataset before 2026-07-21). Always check the regen log's mid-conversation counter before assuming the signal exists. `--split-episode-rate` (with `--split-min-part`) creates natural-continuation-across-sleep signal from multi-turn corpora; single-QA episodes only split (`--split-qa-rate`) when they carry a `question_offsets` entry from `prepare_data.py` (babilong yes, LongAlign never — fail closed); babilong's question+answer tail is only 8–12 tokens (the document is thousands), so any length floor applied to *both* sides of a split-QA cut disqualifies the entire corpus at once — a zero split-tail counter in a regen log means a constraint reached the tail, not bad luck; `--sentence-sleep-rate` reaches inside single-QA document turns via sentence boundaries (built 2026-07-21, deliberately unused until the 396-reproduction question is settled — see `notes/DISCUSSION-20260721-peak-reproducibility.md`).

## Rented-CUDA box gotchas (found 2026-08-06, A10)

- **First `make test` on a fresh instance looks hung.** `tests/test_mixer_fused.py`'s
  third test triggers a cold Triton compile of the SSD backward kernel; `ptxas`
  runs at ~90% CPU for several minutes with pytest printing nothing, which reads
  exactly like a segfault. Check for a live `triton/backends/nvidia/bin/ptxas`
  child before debugging. Warm, the whole suite is ~2 min.
- **Don't detect command completion with `tmux display-message -p '#{pane_current_command}'`** —
  it reports `bash` while `uv run python` is working, so a completion poll (and
  the launch skill's monitor snippet) fires early. Use a file marker
  (`cmd > log 2>&1; echo "EXIT=$?" >> log`) and poll for that.
- **Running GPU jobs in parallel on one GPU is a net loss.** Three concurrent
  `dream_sleep.py` streams managed 0.11 optimizer step/s each (0.33 aggregate)
  against **1.4 step/s for a single job** — ~4x worse in total throughput. The
  low single-job utilisation number (~20%) is not usable headroom: these loops
  are one-token-at-a-time and latency-bound. Run grids serially **on that
  card** — this is an A10 capacity limit, not a property of these loops. On a
  GH200 (measured 2026-08-07) the same three streams cost nothing: one cell
  uses 3.4 GB of 97 GB at 12% utilisation, and three concurrent
  `dream_sleep.py` processes hold their solo step rate (2.2 step/s replay,
  against 2.0 solo) at 99% utilisation — ~2.8x aggregate. Match the decision
  to the card: measure one cell's utilisation before choosing.
- **`pkill -f <script>.py` kills your own shell**, because the tool's command
  string contains the pattern. Kill the tmux window instead.
