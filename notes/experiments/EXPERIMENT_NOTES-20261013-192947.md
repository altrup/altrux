# Experiment notes — 2026-10-13 19:29:47 UTC — LAMA-CKL fourth box (ultra20k-v4)

Branch `box/20261013-185224`. GH200 480GB (97.9 GB VRAM), 64 CPU, 525 GB RAM.
Governing note: DISCUSSION-20261013 (third box debrief). Session soft cap 20 h
from session start (19:29 UTC, cap at 2026-10-14 15:29 UTC).

## State at start

- Nothing prepared: no `sft/data`, no `.cache/LAMA`, no HF cache, no checkpoints.
  Uploaded: `.cache/lama_ckl/upstream/finetune_qlora.pkl` (Llama-2 calibration, reused), `.cache/wheels`.
- `warmstart.sha256` reads `unset` (correct: split guard active until the pin).
- `sft/uv.lock` shows a 198-line local diff left by setup (not mine; left uncommitted).
- torch 2.11.0+cu128, CUDA visible.

## Session log

- 19:29 step 1, warm start, in `train`:
  ```
  cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start
  ```
  (target runs preparation.dolly, preparation.conversations `--hf-dataset HuggingFaceH4/ultrachat_200k --max-examples 20000 --input data/dolly_context.jsonl --max-len 8192 --pack --repeat-rate 0.3 --train-user-turns --output data/warm_start.pt`, then training.cli `--lr 1e-4 --chunk-len 512 --lora-rank 16 --lora-alpha 32 --lora-dropout 0 --data data/warm_start.pt --epochs 1 --ckpt-every-tokens 2600000 --keep-ckpts 11`, then finalize into `checkpoints/ultra20k-v4/`.)
- 19:29 LAMA archive download in `work:lama` (wget dl.fbaipublicfiles.com/LAMA/data.zip, unzip to `.cache/LAMA/data`).
