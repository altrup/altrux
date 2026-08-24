# Experiment notes — 2026-08-23: LAMA-CKL rented-GPU handoff

## Status

Local implementation and CPU/fake-model gates are complete. No rented GPU has
been started. Do not start a full Mamba cell until the upstream Llama gate, the
Mamba split, and all four rented-GPU smokes below pass.
For this rental, this handoff and its linked discussion protocol supersede the
experimenter skill's generic neural-memory exploration goal.

Implementation commits, in order:

- `bd5ddbf` — pinned TAALM release verifier and Llama reproduction gate;
- `f9cdc63` — official-notebook source conversion, Mamba metric, and 500/500
  split builder;
- `18b5041` — conversational wake, EOC closure, instructed dream transition,
  and dream diagnostics;
- `e74c100` — native evidence-document training epoch;
- `f921c3d` — resumable four-arm runtime, compact dreams, and aggregate report;
- `dcc732a` — configurable symmetric cache transfer and selected ARM64 wheels;
- `7aead01` — watchdog coverage for TAALM and LAMA-CKL processes;
- `18b74c9` — upstream result archival under `.cache/lama_ckl/`;
- `b077e40` — one-GH200 batch-equivalent upstream adapter and one-update smoke.

Local evidence:

- `make test`: 633 passed, 5 skipped, 0 failed in 214.11 seconds;
- LAMA-CKL group: 25 passed in 8.04 seconds;
- pinned released artifacts: 500/500/500/4166 rows, all four SHA-256 values
  matched, with zero malformed rows, duplicate UUIDs, or missing bindings;
- tokenizer smoke: separately assembled wake tokens exactly matched the whole
  trained chat rendering, and the dream seed decoded as the registered user
  instruction followed by the assistant marker.

The five skipped tests need the CUDA fused kernel. The local ROCm card cannot
run the paid-run smoke because the runtime deliberately rejects ROCm.

## Rented-machine order

Use the configured GH200: one 96-GiB GPU and 432 GiB host RAM. Install the
project with `TORCH_BACKEND=cu128 make sync`; do not reuse the isolated TAALM
environment for Altrux. The upstream gate is explicitly a single-GH200
batch-equivalent replication: microbatch 8 and accumulation 8 preserve the
released effective batch 64 and optimizer updates per epoch, but not eight-GPU
DDP numerics. Do not describe it as an exact hardware reproduction.

1. From the repository root, prepare the isolated pinned TAALM environment:

   ```bash
   git clone https://github.com/ybseo-ac/TAALM.git .cache/TAALM
   git -C .cache/TAALM checkout --detach b12f344a9dbae555c239635b1c192c555bed001b
   uv venv --python 3.10 .cache/TAALM/.venv
   UV_TORCH_BACKEND=cu128 uv pip install \
     --python .cache/TAALM/.venv/bin/python \
     -r .cache/TAALM/requirements.txt
   sudo install -d -o "$(id -u)" -g "$(id -g)" /results/lamackl
   cd sft
   ```

   The repository has no declared license, so keep the checkout and results
   local. Stop if its pinned dependencies do not install on ARM64; do not
   substitute newer packages during the registered gate.
2. Run `make lama-ckl-upstream-check`.
3. In the `train` tmux, run the 64-row, one-optimizer-update ARM64 dependency
   and training gate with
   `PATH="../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-smoke`.
   Stop on an import, model-load, forward, backward, optimizer, CUDA, or output
   failure. Do not repair it by changing TAALM data or training semantics.
4. In the `train` tmux, run the authors' Llama-2-7B QLoRA cell through the
   registered topology adapter with
   `PATH="../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-run`.
5. In the `work` tmux, run
   `make lama-ckl-upstream-summarize RESULT=/results/lamackl/finetune_qlora.pkl`.
   Stop unless it passes peak acquisition `0.115 ± 0.02`, first peak epoch
   `16 ± 2`, and paired retention `0.8174 ± 0.02`. A passing summary copies
   the result to `.cache/lama_ckl/upstream/finetune_qlora.pkl` for pull-back.
   Run `touch ../scripts/.watchdog-fetch` and verify `../scripts/.pull-receipt`.
6. In the `work` tmux, download and unpack the official LAMA archive with the
   commands in `sft/README.md`.
   Confirm `relations.jsonl` and `TREx/` are directly below `.cache/LAMA`.
7. In the `work` tmux, run
   `make lama-ckl-split ARGS="--model-name mamba2_2_7b --lama-root ../.cache/LAMA"`.
   Read the printed source sample and both final samples. Stop unless both
   artifacts have 500 rows and every printed structural invariant is zero.
   Request and verify another artifact pull.
8. In the `train` tmux, run the four `make lama-ckl-smoke` commands in
   `sft/README.md`. Record rate,
   ETA, peak VRAM, effective batch sizes, EOC termination, assistant EOS, and
   host RAM. Stop on an OOM, malformed wake, duplicate dream, non-finite loss,
   missing decoded sample, or hash mismatch. Fix only technical failures; do
   not change dream content or benchmark scores.
9. Lock the smoke-proven `--dream-batch-size` and `--eval-batch-size`, estimate
   full cost from the logs, then run the 12 registered arm/seed cells serially
   in the `train` tmux.
10. In the `work` tmux, run `make lama-ckl-report`. It must accept exactly the
    four arms at seeds 42, 43, and 44 on one split and one shared protocol.

The TAALM model and Llama weights are gated downloads. Their access tokens are
external prerequisites, not repository configuration. No token belongs in a
command, note, log, or committed file.

## Artifact contract

`.cache/lama_ckl/mamba2_2_7b_recap050/manifest.json` freezes the split source,
pipeline revision, warm-start hash, tokenizer alignment, settings, hashes, and
samples. Each arm writes atomically completed `cycle-NN/` directories under
`.cache/lama_ckl/runs/` and resumes from the latest complete cycle.

Altrux stores dream IDs, decoded text, generation seeds, stop reasons, set hash,
teacher adapter hash, topology, and diagnostics before training. Full teacher
logits stay in memory only through the cycle's distillation. Per-cycle adapters
and the exact dream tokens reconstruct them. Dream diagnostics are read-only;
they cannot select, regenerate, stop, or tune dreams.
