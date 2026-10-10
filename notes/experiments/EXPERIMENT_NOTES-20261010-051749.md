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
