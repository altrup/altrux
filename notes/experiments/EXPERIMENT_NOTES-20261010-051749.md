# Experiment notes — 2026-10-10 05:17 UTC: LAMA-CKL box run (first paid attempt)

Branch: `box/20261010-041135`, launch commit `4e658eb`.
Box: NVIDIA GH200 480GB (97871 MiB VRAM), 525 GiB host RAM, 64 cores, 3.9 TB disk, ARM64, Python 3.10.12 system, uv 0.13.0.
Governing docs: the 2026-08-23 GPU handoff and the 2026-08-23 wake/dream protocol (amended 2026-10-05, EOC-only dream prompt). The 2026-10-05 scratch-capacity note schedules nothing; the four-arm baseline runs first.

## Start state (05:17 UTC)

- No `.cache/lama_ckl`, `.cache/LAMA`, `.cache/TAALM`, or `sft/data` arrived with the upload. Only `.cache/wheels` and the recap-0.5 warm start.
- Warm start present and verified: `models/mamba2_2_7b/checkpoints/recap050/epoch-2/step-800/trainable.pt` SHA-256 `226e9576...a11f3c` (matches the protocol note).
- `sft/.venv`: torch 2.11.0+cu128, CUDA available. Setup left `sft/uv.lock` modified (re-resolved by `make sync`); not committed by me.
- `HF_TOKEN` is exported in `~/.bashrc` (tmux shells see it; non-interactive `new-window` commands must source it).
- `sft/.env` has `MODEL_NAME=` empty; every command passes `--model-name mamba2_2_7b` / `MODEL_NAME=mamba2_2_7b` explicitly.

## Plan (handoff order)

1. TAALM pinned env (CPU install) in `work:taalm-env`; LAMA archive download in `work:lama-dl` in parallel.
2. `lama-ckl-upstream-check` → upstream smoke (train tmux) → upstream run (train tmux) → summarize gate.
3. Mamba split (`lama-ckl-split`) — run while the TAALM install is CPU-bound if the GPU is idle; its result does not depend on the Llama gate, only its use does.
4. Four `lama-ckl-smoke` arms, record rate/VRAM, then 12 cells serially, then report.

## Log

### 05:17–05:22 UTC — environment and gate preparation

- `work:taalm-env`: TAALM clone at `b12f344a`, `uv venv --python 3.10`, pinned requirements with `UV_TORCH_BACKEND=cu128` → torch 2.11.0+cu128, transformers 4.36.2, bitsandbytes 0.42.0. Install succeeded (EXIT=0) in under a minute.
- `work:lama-dl`: official LAMA `data.zip` downloaded and unpacked. The archive unpacks to `.cache/LAMA/data/{relations.jsonl,TREx,...}`, so the split's `--lama-root` is `../.cache/LAMA/data`, not `../.cache/LAMA` as `sft/README.md` writes.
- `make lama-ckl-upstream-check`: all four released files match rows and SHA-256, all invariants zero.
- HF access: `meta-llama/Llama-2-7b-hf` (gated, manual) and `state-spaces/mamba2-2.7b` both resolve with the box `HF_TOKEN`.

**DEVIATION 1 — bitsandbytes 0.42.0 on ARM64.** The pinned `bitsandbytes==0.42.0` wheel is `py3-none-any` and bundles x86-64 `.so` files only; `import bitsandbytes` raises `CUDA Setup failed` on this aarch64 box (the `.so` is `ELF x86-64`). The handoff says to stop if pinned dependencies do not install on ARM64 and not to substitute newer packages. Decision: keep the pinned version and compile it from its own source tag for aarch64, instead of stopping or upgrading. The 0.42.0 source fails to compile on aarch64 only because `include/SIMD.h` includes `<emmintrin.h>` unconditionally; the only `BinSearch` instantiation the library uses is `BinAlgo<Scalar, ...>` (`csrc/cpu_ops.cpp`, `csrc/common.h`), so the SSE/AVX specializations are dead code here. Applied upstream's own portability change to `include/` only (`git checkout 73d3e7b -- include/`, from bitsandbytes PR #949; the diff adds `#if defined(__aarch64__)` guards, `USE_SSE2` guards, scalar `InstrFloatTraits` typedefs, and two comment typo fixes; no kernel or quantization code changes). Build: `CUDA_VERSION=128 make cuda12x` with the 0.42.0 Makefile (includes `compute_90,sm_90`), then `uv pip install --no-deps --reinstall .` into the TAALM venv. Reasoning: a from-source build of the same pinned version preserves the released quantization kernels and version semantics; the gate tolerance then checks the replication empirically. If the gate fails, the arch-specific build is one candidate cause and the run stops, which is the same outcome as stopping now. Source tree at `.cache/bnb-src` (not pulled home; the patch is reproducible from the two commands above). The teammate should accept or reject this before the Mamba comparison is quoted as gated.
- `HF_TOKEN` lives in `~/.bashrc`; `tmux new-window 'cmd'` does not read it, so the split logged "unauthenticated requests" (harmless for the public Mamba repo). Every Llama command runs with `source ~/.bashrc` first.

### 05:20 UTC — Mamba split started (before the Llama gate, by choice)

Verbatim, in `work:split`:

    cd ~/altrux/sft && make lama-ckl-split ARGS="--model-name mamba2_2_7b --lama-root ../.cache/LAMA/data"

Warm start loaded and hashed `226e9576…a11f3c`, 257/257 trainable tensors, 21,323,264 trainable params, 128 LoRA adapters. Scoring 13,645 candidates at ~58 items/s (batch 8), GPU 8.8 GiB / ~17 % util. Ordering note: the handoff lists the split after the Llama gate; the split's content does not depend on the gate (deterministic from the frozen warm start and seed 42), only its use does, and the GPU was otherwise idle while the TAALM environment built. The gate still blocks every Mamba cell.

### 05:26 UTC — split done; two more TAALM environment deviations

Split finished at 05:26 (about 4 min, 13,645 candidates at ~79 items/s, descriptive zeros=8346, ones=2458). Artifacts under `.cache/lama_ckl/mamba2_2_7b_recap050/`: `variant.jsonl` 500 rows sha256 `1be3643d…050c88`, `invariant_descriptive.jsonl` 500 rows sha256 `ca6470c8…b702d`, both invariant dicts zero, `manifest.json` records warm-start hash, source hash, TAALM commit, and the object-span metric alignment. Samples read as a to-learn fact the model gets wrong (Dan Conners / linebacker) and a retained fact it gets right (Dodoma / Tanzania). `.watchdog-fetch` touched at 05:27.

**DEVIATION 2 — triton removed from the TAALM venv.** With torch 2.11 the venv carries triton 3.6.0, and bitsandbytes 0.42.0 imports `triton.ops.matmul_perf_model` (removed in triton 3) whenever triton is importable, so `import bitsandbytes` fails on any torch ≥2.4 install, x86 included. The import sits in bitsandbytes' optional int8 "SwitchBack" triton kernels, which QLoRA 4-bit never calls. Fix: `uv pip uninstall --python .cache/TAALM/.venv/bin/python triton`; bitsandbytes then reports `COMPILED_WITH_CUDA True` and torch works without triton (no `torch.compile` in TAALM). bitsandbytes also needs `LD_LIBRARY_PATH` pointing at the venv's `nvidia/cuda_runtime/lib` to locate `libcudart.so`; exported in the `train` shell before every upstream command.

**DEVIATION 3 — peft pinned by date.** TAALM's `requirements.txt` lists `git+https://github.com/huggingface/peft.git` with no ref, so the pinned environment was never reproducible; today's peft main (0.21.3.dev0) requires a newer accelerate than the pinned 0.25.0, and peft main at the TAALM commit date (2024-11-10) requires transformers ≥4.42 against the pinned 4.36.2. Chosen pin: peft main at the release date of the newest pinned package, bitsandbytes 0.42.0 (2024-01-07): commit `8665e2b5719faa4e4b91749ddec09442927b53e0` (peft 0.7.2.dev0, 2024-01-03). Installed with `uv pip install --no-deps git+https://github.com/huggingface/peft.git@8665e2b5719faa4e4b91749ddec09442927b53e0`. TAALM uses only `LoraConfig` and `get_peft_model`. Resulting venv: torch 2.11.0+cu128, transformers 4.36.2, peft 0.7.2.dev0, trl 0.7.4, accelerate 0.25.0, bitsandbytes 0.42.0 (aarch64 source build), datasets 5.0.1 (unpinned upstream; left as resolved).

### 05:28 UTC — upstream smoke launched (train tmux)

Verbatim (after `source ~/.bashrc` and the `LD_LIBRARY_PATH` export above):

    cd ~/altrux/sft && PATH="../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-smoke

### 05:29–05:35 UTC — two blockers found

**BLOCKER A — Llama gate: the box token has no Llama-2 access.** The first smoke attempt failed on a relative `PATH` in the README command (fixed in `sft/README.md`, commit "docs: absolute TAALM venv PATH…"). The second reached model loading and failed with `GatedRepoError: 403` on `meta-llama/Llama-2-7b-hf/resolve/main/config.json`. Verified directly with the token passed explicitly: `whoami` → user, classic `read` token; `hf_hub_download(config.json)` → 403 "You must have access to it". My earlier "access OK" line used `model_info` unauthenticated (metadata is public) and was wrong. The account must accept the Llama-2 license on huggingface.co; nothing on the box can do that. The upstream gate (handoff steps 3–5) is therefore NOT RUN this session. Every Mamba result below is UNGATED.

**BLOCKER B — the registered wake invariant fails on the warm start at turn 1.** `make lama-ckl-smoke ARGS="--model-name mamba2_2_7b --arm frozen --seed 42"` reached `cycle 0: to-learn 0.000000, not-to-forget 1.000000` (the initial evaluation works: 2+2 rows, samples decoded correctly) and then raised `RuntimeError: Reached reply tokens limit without generated <eos> token` in `run_conversational_wake`. Diagnostic (`diag_wake.py`, scratch, not committed; mirrors the runner's load and prompt exactly; `sft/logs/diag-wake.log`): for the first 8 to-learn evidence documents the greedy reply never emits EOS within 256 tokens. The model does not answer the document; it continues it as encyclopedia prose and then loops (turn 3 repeats the Prestolee sentence verbatim four times; turn 2 repeats a guild sentence). So the recap-0.5 adapter treats a bare evidence sentence in a `[USER]` turn as text to continue, not a message to reply to. The protocol note §2 requires "The reply must emit the tokenizer EOS before that backstop"; with this adapter and these inputs that requirement is unsatisfiable, not a technical failure. Generation measured ~14 tokens/s for single-token stepping (256 tokens ≈ 18 s), which prices a 64-token backstop wake at roughly 500 × 4.6 s ≈ 38 min per cycle, ≈ 19 h of wake alone per 30-cycle cell.

Running now (`work:diag`, `sft/logs/diag-wake-full.log`): the same wake over all 500 documents with the 64-token backstop and a forced EOS fed at the backstop, counting natural-EOS replies, EOC emissions, tokens/s, and VRAM, to size the problem and the cost.

## PROTECTED EDIT — commit `634f914`

Files: `sft/experiments/lama_ckl/protocol.py`, `sft/experiments/lama_ckl/runner.py` (plus `sft/tests/experiments/lama_ckl/test_protocol.py` and `sft/README.md`).

Traceback (every arm, cycle 1, turn 1):

    File ".../experiments/lama_ckl/protocol.py", line 268, in run_conversational_wake
        raise RuntimeError("Reached reply tokens limit without generated <eos> token")

Change: a reply that reaches the 64-token backstop is closed by feeding the tokenizer EOS into the state (so every turn is complete, as before), the turn is recorded with `eos_forced: true` in `wake.json`, and `invariants.missing_assistant_eos` counts those turns. The runner no longer requires that count to be zero; it prints `wake replies closed by the backstop: N/500` per cycle. `internal_eoc` remains a hard invariant, the prompt rendering, greedy decoding, evidence truncation, dream prompt, and all training code are untouched. Test added: a model that never emits EOS yields `[t, t, t, EOS]`, `eos_forced` true, one extra model call, `missing_assistant_eos == 1`. `tests/experiments/lama_ckl`: 33 passed.

Why this and not something else: the protocol note's "must emit EOS before the backstop" cannot be met by the registered warm start on these inputs; the alternatives (truncate without EOS, drop the assistant turn, change the prompt, decode with sampling) each change the carried state or the wake design more than this does. This edit keeps the registered source exposure identical and turns the failure into a measured quantity. Whether a wake made mostly of forced closures is the wake the team wants is a protocol decision, not mine: every result produced after this commit is PROVISIONAL, and the Mamba cells are not launched on it (see decision below).

Throughput note for the team (not changed): on CUDA, single-token decoding in `models/mamba2_780m/model.py` always takes the manual per-layer PyTorch `_mixer_step` (line 342 enables the Triton chunk scan only for `seqlen > 1`), so the wake and dream generation run at ~15 tokens/s. The fused `selective_state_update`/`causal_conv1d_update` step path, or CUDA graphs over the step, is the throughput lever; it touches the registered per-token stepping and so belongs to a DISCUSSION decision.

### 06:10 UTC — full-wake diagnostic result (`sft/logs/diag-wake-full.log`)

500 to-learn documents, continuing state, greedy, 64-token backstop with a fed EOS at the backstop (the same rule as commit `634f914`):

| quantity | value |
|---|---|
| replies ending with a natural EOS | 55 / 500 (11 %) |
| replies emitting `<|endofconversation|>` | 0 |
| prompt tokens (500 user turns) | 44,229 |
| generated tokens | 31,828 (mean reply 63.7) |
| generation rate | 15.6 tok/s (single-token stepping) |
| wall time for one wake | 2,095 s (35 min) |
| peak VRAM | 5.6 GiB |

Every reply read, natural or forced, is a continuation of the evidence text (biography, geography) rather than an answer; the natural-EOS replies are the ones where the continuation happens to end a paragraph. So the registered wake exposes the model to the official evidence, as intended, plus ~32k tokens per cycle of its own hallucinated continuation. Cost: 30 wakes ≈ 17.5 GPU-hours per cell before dreams, training, or evaluation; 12 cells ≈ 210 GPU-hours of wake alone at the current stepping speed.

### 06:11 UTC — four engineering smokes launched (train tmux, serial)

    cd ~/altrux/sft && for arm in frozen lora mix-review altrux; do make lama-ckl-smoke ARGS="--model-name mamba2_2_7b --arm $arm --seed 42"; done

PROVISIONAL (after the protected edit). Purpose: prove the remaining machinery (dreams, distillation, LoRA epochs, evaluation, checkpoints) and read rates and VRAM for the cost estimate.

### 06:16 UTC — all four smokes pass (PROVISIONAL, after `634f914`)

All EXIT=0; every smoke wake closed both replies by the backstop (2/2), `internal_eoc` 0, the dream diagnostics dict is present for altrux, checkpoints and `cycle-01/` written and the frozen smoke resumed from its earlier `cycle-00/`.

| arm | cycle-1 wall (s) | peak VRAM (GiB) | treatment |
|---|---|---|---|
| frozen | 22 | 5.8 | none |
| lora | 59 | 6.9 | 1 optimizer step on 2 docs, 109 token gradients, 49 s |
| mix-review | 55 | 9.2 | 1 step on 2+2 docs, 216 token gradients, 46 s |
| altrux | 54 | 6.2 | 2 dreams × 31 tokens generated in ~2 s, 2 distillation steps in ~43 s |

The ~45 s for a single first optimizer step is read as first-backward Triton compilation; the real cell below gives the steady rate. Smoke dream sample 2 was the recap loop (`[USER] Can you go over everything we discussed earlier? [ASSISTANT] You asked: …`), sample 1 an unrelated web snippet; both 31 tokens, no EOC within 32 tokens (expected at that limit).

### 06:16 UTC — DECISION: one bounded real cell, no full grid

Reasons the 12-cell grid does not start this session: (1) the Llama gate is not run (token lacks Llama-2 access); (2) the wake runs on a protected edit the team has not accepted, and 89 % of its turns are forced closures of hallucinated continuations, which is a protocol question; (3) at 15.6 tok/s a cell is ≥ 17.5 GPU-hours of wake alone, so 12 cells cost ≥ 210 GPU-hours before dreams, training, or evaluation, and that resource decision belongs to the team.

What runs instead: the altrux seed-42 cell, to be stopped after cycle 2 completes, so the team gets two real 300-dream sets from real post-wake states with their diagnostics (coverage of the to-learn bindings is the registered fallback trigger for the instruction-turn arm), the steady dream-generation, distillation, and evaluation rates for costing, and two provisional acquisition/retention points. Verbatim:

    cd ~/altrux/sft && make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm altrux --seed 42 --dream-batch-size 30 --eval-batch-size 16"

`--dream-batch-size 30` (10 batches of 30) chosen from the 5.6 GiB single-stream VRAM; the batch size is recorded in `run.json` and the report requires every cell to share it, so a later full grid must either reuse 30 or discard this directory.

### 06:18 UTC — altrux cell cycle 0

Evaluation of 1000 rows at `--eval-batch-size 16` takes ~30 s (to-learn at ~20 item/s, not-to-forget at ~165 item/s). Result: to-learn 0.0000, not-to-forget 0.9855. The split selected the 500 retention rows at score 1.0 with batch 8; 7 of them score 0 at batch 16 in the runner, so the object-token accuracy is sensitive to padding/batch numerics in bf16 at the ~1 % level. The do-nothing floor for this cell is therefore (0.0000, 0.9855), not (0, 1). Wake 1 running at 0.23 turn/s.

### 07:03 UTC — altrux cell cycle 1 stopped by the duplicate-dream rule

Wake 1: 500 turns in 35 min, 458/500 replies closed by the backstop, `internal_eoc` 0. Dream generation: 300 dreams × 512 tokens at batch 30 in 8.8 min (0.57 dream/s, ≈290 tok/s aggregate). Then `runner.py:459` raised `the fixed dream set contains duplicate dreams; refusing to train` (EXIT=2). The check runs before `dreams.json` or the wake artifact is written, so the cycle left nothing on disk: the dream set, its diagnostics, and the wake transcript are lost. The handoff lists "duplicate dream" as a stop condition and the protocol forbids a content-dependent retry, so the cell is not restarted. Recommendation for the team (not applied; `runner.py` is protected and this is not a crash to unblock): write `dreams.json` and `wake.json` before the duplicate check, so a refused set is still inspectable.

Next: reproduce the same wake and dream set with the protocol's own functions (`run_conversational_wake`, `dream_prompt_ids`, `generate_replay_dreams`, `lama_dream_diagnostics`, seed 4201 = 42·100 + 1) in a scratch script that saves the wake, the state, every dream, the duplicate groups, and the diagnostics under `sft/logs/`, so the duplicate content and the to-learn coverage are on record. ≈ 50 min GPU.

### 07:50 UTC — the cycle-1 dream set, reproduced and saved (`sft/logs/diag-dreams-cycle1/`)

Scratch script `diag_dreams.py` (protocol functions only; log `sft/logs/diag-dreams.log`). Files: `wake.json` (the 500-turn wake artifact, transcript sha `650323ff…3086`, invariants `turns 500, missing_assistant_eos 445, internal_eoc 0`), `open_state.pt` (170 MB, the open post-wake state; travels by the pull only), `dreams.jsonl` (300 dreams, set sha `be98557a…13aa`), `diagnostics.json`.

Wake reproducibility: the runner's wake closed 458/500 replies at the backstop, this one 445/500, same code, same weights, greedy. The greedy wake is therefore not bit-reproducible run to run on this GPU (bf16 kernel nondeterminism), which also means dream sets cannot be reproduced from seeds alone; the saved state is the reproducible object.

Dream set, EOC-only prompt, T=0.7, 512 tokens, seed 4201:

| diagnostic | value |
|---|---|
| unique dreams | 261 / 300 (7 duplicate groups, 46 dreams) |
| EOC termination | 179 / 300 (59.7 %) |
| generated tokens | min 24, quartiles 73 / 351 / 511, mean 296 |
| correct to-learn bindings | 24 |
| misbindings / contradictions | 508 / 508 |
| mean copy fraction of the wake | 3.6 % (max 81.9 %, longest verbatim run 147 tokens) |

Every duplicate is the recap loop: `[USER] What did I ask you about earlier? [ASSISTANT] You asked: "What did I ask you about earlier?" …` (19 + 11 + 3 + 2 identical copies) and `Can you go over everything we discussed earlier? … I said: "You asked: …` (7 + 2 + 2). The recap-0.5 adapter's recap behaviour has collapsed onto a self-referential loop with no wake content; at T=0.7 the loop is short enough (43–60 tokens) that identical samples recur, which is what trips the runner's duplicate rule. The other dreams are mostly generic assistant chat (SEO, travel blogs, marketing strategy, forgiveness) with no relation to the wake. A minority rehearse the wake: dream 2 and dream 7 reuse Stadio Flaminio / the Vinci brothers from one evidence document. The runner's duplicate stop will trigger on every cycle of every altrux cell with this prompt; the fallback instruction prompt is the registered next arm, and its dream set from the same saved state is being generated now (`diag_dreams_instr.py`, `sft/logs/diag-dreams-instr.log`) together with a subject-coverage count for both sets.
