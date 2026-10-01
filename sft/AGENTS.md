# SFT instructions

## Always update the README

Whenever you change user-facing behaviour — new CLI flags, changed defaults, a removed feature — update `sft/README.md` in the same change. Keep the usage examples and key flags accurate; point users to `--help` for the full option list.

Whenever you add, rename, or remove an environment variable, also update `sft/.env.example`.

## Every module names its owning experiment

The first line of each module docstring under `sft/` and `models/` (not tests, not `__init__.py`) is `Experiment: <slug>`, with slug one of `lama-ckl`, `state-erasure`, `memory-model`, `shared`. `shared` means more than one experiment, or generic training/preparation infra, reaches the module. The header records ownership only: `notes/README.md` decides which experiments are live. `tests/test_experiment_headers.py` enforces it and skips `PROTECTED_PATHS` files, whose headers the person adds by hand.

## Import order is load-bearing for `models`

`models/__init__.py` installs the `selective_scan_cuda` stub that every
`mamba_ssm` import on this box depends on, so `import models` must come before
any `mamba_ssm` import in a file. `make fmt` sorts imports with ruff, and
`pyproject.toml` gives `models` its own isort section ahead of third-party so
the sorter keeps that order instead of breaking it. Don't move `models` back
into first-party.

## Cache policy

HuggingFace cache is shared at the repo root (`../.cache/huggingface`), not inside this folder. The Makefile sets `HF_HOME=$(CURDIR)/../.cache/huggingface` on every relevant target. See the root `AGENTS.md` for the full policy.

uv uses its default system cache — no override needed.

## Model configuration

The model is selected via `MODEL_NAME` in `.env`. This controls which package in `models/` is imported. The scripts add the repo root to `sys.path` automatically; the Makefile also sets `PYTHONPATH`. See the root `AGENTS.md` for the full model interface contract.

## Checkpoint format

`state.pt` (written by `training/checkpoints.py`) carries a `dataset_fingerprint` (resolved `--data` path + example count) alongside `slot_states`/`next_ptr`, so `training/cli.py`'s resume path can tell whether `--data` still points at the dataset those indices were recorded against and reset them instead of silently reindexing into an unrelated dataset. If you touch `state.pt`'s schema again, keep this field — `slot_states`/`next_ptr` are meaningless without it.

## No spaCy in this venv

`preparation/cram.py` needs NER and the next-run plan called for spaCy, but this venv is Python 3.14 and spaCy publishes no cp314 wheels — `uv pip install spacy` resolves, then source-builds thinc/blis/murmurhash/preshed/cymem/srsly, on top of the standing risk that any `uv` install re-resolves torch off the ROCm build (root `AGENTS.md`). It uses `transformers`' CoNLL-03 token classifier instead (already a dependency, PER/ORG/LOC is all a same-type entity swap needs). Don't re-attempt spaCy without checking for cp314 wheels first.

## Data-shape gotcha: mid-conversation sleeps need episodes that don't exist

`--sleep-chain-rate` (0.1) gates *all* of the below: an episode's rates only apply if its chain drew into the sleeping fraction, so every sleep/split/fact count in a regen log is ~10% of what the per-episode rates alone imply. Then `preparation/chains.py --mid-sleep-rate` only fires on episodes ≥ `--mid-sleep-min-len` (4096 default; 1536 used in practice) that have a turn boundary in their middle third. LongAlign/babilong episodes are single-QA (turn boundaries only at position 0 and the answer start) so they never qualify; ultrachat only qualifies when tokenized uncapped (`preparation/conversations.py --max-len`, default 32768 — the old 1024 cap made the set empty and mid-sleep a silent no-op in every dataset before 2026-07-21). Always check the regen log's mid-conversation counter before assuming the signal exists. `--split-episode-rate` (with `--split-min-part`) creates natural-continuation-across-sleep signal from multi-turn corpora; single-QA episodes only split (`--split-qa-rate`) when they carry a `question_offsets` entry from `preparation/conversations.py` (babilong yes, LongAlign never — fail closed); babilong's question+answer tail is only 8–12 tokens (the document is thousands), so any length floor applied to *both* sides of a split-QA cut disqualifies the entire corpus at once — a zero split-tail counter in a regen log means a constraint reached the tail, not bad luck; `--sentence-sleep-rate` reaches inside single-QA document turns via sentence boundaries (built 2026-07-21, deliberately unused until the 396-reproduction question is settled — see `notes/discussion/DISCUSSION-20260721-peak-reproducibility.md`).

## The local datasets cache has a stray file where ultrachat's dir belongs

`.cache/huggingface/datasets/HuggingFaceH4___ultrachat_200k` is a 96-byte
*file* on this box, so any `load_dataset("HuggingFaceH4/ultrachat_200k")` dies
with `NotADirectoryError` before it downloads anything (hit by
`experiments/dreams/cli.py --wake-dialogue`). Remove that file to let the cache directory
be created; a fresh box has no such artifact.

## Rented-CUDA box gotchas (found 2026-08-06, A10)

- **First `make test` on a fresh instance looks hung.** `tests/experiments/dreams/test_mixer_fused.py`'s third test triggers a cold Triton compile of the SSD backward kernel; `ptxas` runs at ~90% CPU for several minutes with pytest printing nothing, which reads exactly like a segfault. Check for a live `triton/backends/nvidia/bin/ptxas` child before debugging. Warm, the whole suite is ~2 min.
- **Don't detect command completion with `tmux display-message -p '#{pane_current_command}'`** — it reports `bash` while `uv run python` is working, so a completion poll (and the launch skill's monitor snippet) fires early. Use a file marker (`cmd > log 2>&1; echo "EXIT=$?" >> log`) and poll for that.
- **Grid concurrency is a per-card decision — measure one cell's utilisation before choosing; neither "serial" nor "parallel" is a standing rule.** A10: three concurrent `make dream-sleep` streams managed 0.11 optimizer step/s each (0.33 aggregate) against **1.4 step/s for a single job** — ~4x worse; the low single-job utilisation (~20%) is not usable headroom (per-token latency-bound loops). Run grids serially on an A10. GH200 (measured 2026-08-07): the same three streams cost nothing — one cell uses 3.4 GB of 97 GB at 12% utilisation, and three concurrent processes hold their solo step rate (2.2 step/s replay vs 2.0 solo) at 99% utilisation — ~2.8x aggregate. Run grids concurrently on a GH200 (per-token arms still saturate ~2.5 aggregate step/s — launch-latency-bound; see the batching rule below).
- **Batching is mandatory for box code (standing direction, altrup 2026-08-11).** Anything that iterates per-token or per-item on a GPU must take a batch dimension — single-stream decode of a 2.7B model on a GH200 measured 13–14 tok/s, launch-latency-bound, and `nvidia-smi`'s 97% "utilization" is misleading there (busy launching kernels, not computing). Independent work (e.g. the N dreams of a set) batches at ~N×; co-scheduling more processes does not substitute for latency-bound loops. Implementation blockers for batched dream generation are enumerated in `notes/experiments/EXPERIMENT_NOTES-20260810-231500.md`.
- **`pkill -f <script>.py` kills your own shell**, because the tool's command string contains the pattern. Kill the tmux window instead.
