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
