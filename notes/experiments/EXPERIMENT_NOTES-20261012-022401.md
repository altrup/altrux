# Experiment notes — 2026-10-12 02:24 UTC — LAMA-CKL third box run

Branch `box/20261012-020915`. GH200 480GB (97.9 GB VRAM visible). Governing note:
`DISCUSSION-20261011-lama-ckl-second-box-debrief.md` (§3.3 session order).
Model `mamba2_2_7b` on every command.

## Arrival inventory (02:21 UTC)

- No `models/mamba2_2_7b/checkpoints/` arrived: the warm start retrains here (as registered).
- `.cache/lama_ckl/mamba2_2_7b_repeat030/` (old split, conditioned on the retired repeat-0.3
  warm start) and `.cache/lama_ckl/upstream/finetune_qlora.pkl` (Llama-2 calibration, kept,
  not rerun per §3.3 step 4) arrived. The old split is not used; a new split is built on
  `ultra20k`.
- `sft/experiments/lama_ckl/warmstart.sha256` arrived holding the old repeat-0.3 pin
  `245e2089aa51d923747030feb8d77f23e691803478269a8bc222b2591cbf70fe`. Reset to `unset` at
  02:24 so the split/runner guard is live until the new warm start is pinned (PROTECTED path;
  committed on the box branch as the brief directs).
- `.cache/LAMA` absent: downloading in `work:lama-dl`. `.cache/TAALM` absent: not needed
  (calibration not rerun).
- `sft/uv.lock` modified by setup's `make sync` (py3.12 re-resolve); torch 2.11.0+cu128,
  CUDA visible. Left uncommitted.
- Old logs from 2026-10-10 in `sft/logs/` arrived with the upload (EXIT markers already
  consumed by the monitor).

## Session log

- 02:22 `train`: `cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start`
  (verbatim; Makefile ARGS empty). Dolly prep started 02:22:24.
- 02:22 `work:lama-dl`: `wget https://dl.fbaipublicfiles.com/LAMA/data.zip` → `.cache/LAMA/data/`.
