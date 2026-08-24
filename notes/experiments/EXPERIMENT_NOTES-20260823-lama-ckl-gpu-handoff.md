# Experiment notes — 2026-08-23: LAMA-CKL rented-GPU handoff

## Status

Local implementation and CPU/fake-model gates are complete. No rented GPU has
been started. Do not start a full Mamba cell until the upstream Llama gate, the
Mamba split, and all four rented-GPU smokes below pass.

Implementation commits, in order:

- `bd5ddbf` — pinned TAALM release verifier and Llama reproduction gate;
- `f9cdc63` — official-notebook source conversion, Mamba metric, and 500/500
  split builder;
- `18b5041` — conversational wake, EOC closure, instructed dream transition,
  and dream diagnostics;
- `e74c100` — native evidence-document training epoch;
- `f921c3d` — resumable four-arm runtime, compact dreams, and aggregate report.

Local evidence:

- `make test`: 629 passed, 5 skipped, 0 failed in 236.60 seconds;
- new LAMA group: 21 passed;
- pinned released artifacts: 500/500/500/4166 rows, all four SHA-256 values
  matched, with zero malformed rows, duplicate UUIDs, or missing bindings;
- tokenizer smoke: separately assembled wake tokens exactly matched the whole
  trained chat rendering, and the dream seed decoded as the registered user
  instruction followed by the assistant marker.

The five skipped tests need the CUDA fused kernel. The local ROCm card cannot
run the paid-run smoke because the runtime deliberately rejects ROCm.

## Rented-machine order

Use a CUDA machine with enough GPU memory for the 2.7B training smoke and at
least 64 GiB host RAM for the in-memory 300-dream teacher logits. Install the
project with the CUDA torch backend through `make sync`; do not reuse the
isolated TAALM environment for Altrux.

1. Clone TAALM into `.cache/TAALM`, detach commit
   `b12f344a9dbae555c239635b1c192c555bed001b`, and install its released
   requirements in `.cache/TAALM/.venv`. The repository has no declared
   license, so keep the checkout and results local.
2. Run `make lama-ckl-upstream-check`.
3. Run the authors' Llama-2-7B QLoRA cell with
   `PATH="../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-run`.
4. Run
   `make lama-ckl-upstream-summarize RESULT=/results/lamackl/finetune_qlora.pkl`.
   Stop unless it passes peak acquisition `0.115 ± 0.02`, first peak epoch
   `16 ± 2`, and paired retention `0.8174 ± 0.02`. A passing summary copies
   the result to `.cache/lama_ckl/upstream/finetune_qlora.pkl` for pull-back.
5. Unpack the official LAMA data into `.cache/LAMA`, with `relations.jsonl`
   and `TREx/` directly below that directory.
6. Run `make lama-ckl-split ARGS="--lama-root ../.cache/LAMA"`. Read the
   printed source sample and both final samples. Stop unless both artifacts
   have 500 rows and every printed structural invariant is zero.
7. Run the four `make lama-ckl-smoke` commands in `sft/README.md`. Record rate,
   ETA, peak VRAM, effective batch sizes, EOC termination, assistant EOS, and
   host RAM. Stop on an OOM, malformed wake, duplicate dream, non-finite loss,
   missing decoded sample, or hash mismatch. Fix only technical failures; do
   not change dream content or benchmark scores.
8. Lock the smoke-proven `--dream-batch-size` and `--eval-batch-size`, estimate
   full cost from the logs, then run the 12 registered arm/seed cells serially.
9. Run `make lama-ckl-report`. It must accept exactly the four arms at seeds
   42, 43, and 44 on one split and one shared protocol.

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
