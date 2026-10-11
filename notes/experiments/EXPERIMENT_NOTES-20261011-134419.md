# Experiment notes — 2026-10-11 13:44:19 UTC: LAMA-CKL second box session (seed-42 grid)

Branch `box/20261011-074114`. Box: NVIDIA GH200 480GB (97,871 MiB VRAM), 64 CPU,
525 GiB RAM, 3.9 TB disk. Governing note:
[DISCUSSION-20261010 debrief](../discussion/DISCUSSION-20261010-lama-ckl-first-box-debrief.md),
session order per its 2026-10-10 amendment: retrain, format acceptance, pin,
split, wake acceptance, smokes, Llama-2 calibration, cells. Soft cap 20 h wall
clock (ASSUMED VALUE in the debrief; session start 13:42 UTC → cap 09:42 UTC
2026-10-12). Model `mamba2_2_7b` on every command.

## Arrival state (13:42 UTC)

- Uploaded: only `.cache/wheels/` (aarch64 mamba_ssm, causal_conv1d). No
  checkpoint (`LAMBDA_RESUME_CHECKPOINT` empty, as the debrief §3.7.8 allows),
  no LAMA archive, no TAALM env, `.cache/lama_ckl/` empty,
  `warmstart.sha256` = `unset`. Everything is regenerated here.
- Venv: torch 2.11.0+cu128, CUDA visible, `import mamba_ssm` OK.
- `HF_TOKEN` is in `~/.bashrc` after the interactive guard, so no
  non-interactive shell sees it; every tmux job exports it explicitly with
  `export $(grep -o 'HF_TOKEN=hf_[A-Za-z0-9_]*' ~/.bashrc)`. With it:
  whoami = Altrup, `meta-llama/Llama-2-7b-hf` config download OK → the
  Llama-2 calibration can run this session (it could not on 2026-10-10).
- Uncommitted `sft/uv.lock` change from `make sync` at setup (known, debrief
  §6); left alone.

## 13:44 UTC — three jobs launched

`train` (verbatim, after the HF_TOKEN export):

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start; echo "EXIT=$?" >> "$(ls -t logs/warm-start-*.log | head -1)"

Log `sft/logs/warm-start-20261011-134419.log`. Fresh start, no prior checkpoint.

`work:lama-dl`: official `data.zip` (74,619,466 bytes) → `.cache/LAMA/data/`,
EXIT=0 at 13:44. Log `sft/logs/lama-dl-20261011-134419.log`.

`work:taalm-env`: script (copied below) reproducing the 2026-10-10 TAALM
environment with its three ARM64 deviations (bitsandbytes 0.42.0 from source
with upstream `include/` guards from 73d3e7b; triton removed; peft pinned to
8665e2b5). Log `sft/logs/taalm-env-20261011-134419.log`.

```bash
#!/bin/bash
# TAALM pinned environment on ARM64 with the three deviations recorded in
# notes/experiments/EXPERIMENT_NOTES-20261010-051749.md.
set -euo pipefail
cd ~/altrux
export $(grep -o 'HF_TOKEN=hf_[A-Za-z0-9_]*' ~/.bashrc)
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
log "clone TAALM"
[ -d .cache/TAALM/.git ] || git clone https://github.com/ybseo-ac/TAALM.git .cache/TAALM
git -C .cache/TAALM checkout --detach b12f344a9dbae555c239635b1c192c555bed001b
log "venv"
uv venv --python 3.10 .cache/TAALM/.venv
PY=.cache/TAALM/.venv/bin/python
log "pinned requirements (cu128)"
UV_TORCH_BACKEND=cu128 uv pip install --python $PY -r .cache/TAALM/requirements.txt
log "DEVIATION 1: bitsandbytes 0.42.0 from source for aarch64"
rm -rf .cache/bnb-src
git clone https://github.com/TimDettmers/bitsandbytes.git .cache/bnb-src
git -C .cache/bnb-src checkout --detach 0.42.0
git -C .cache/bnb-src checkout 73d3e7b -- include/
(cd .cache/bnb-src && CUDA_VERSION=128 make cuda12x)
(cd .cache/bnb-src && uv pip install --python ~/altrux/$PY --no-deps --reinstall .)
log "DEVIATION 2: remove triton"
uv pip uninstall --python $PY triton || true
log "DEVIATION 3: peft pinned by date"
uv pip install --python $PY --no-deps git+https://github.com/huggingface/peft.git@8665e2b5719faa4e4b91749ddec09442927b53e0
log "results dir"
sudo install -d -o "$(id -u)" -g "$(id -g)" /results/lamackl
log "verify imports"
export LD_LIBRARY_PATH=$HOME/altrux/.cache/TAALM/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
$PY -c "import torch, transformers, peft, trl, accelerate, bitsandbytes as b; print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__, accelerate.__version__, b.__version__, b.COMPILED_WITH_CUDA)"
log "DONE"
```

## 13:45–13:52 UTC — warm start attempt 1: stopped on a loss spike, then explained

Packer (log `sft/logs/warm-start-20261011-134419.log`): 710 examples, 2476
conversations, 1766 boundaries (476 repeat: 354 adjacent, 122 gapped; 1290
fresh), 1,926,846 tokens; all five "must be 0" invariants 0; decoded samples
at a fresh, an adjacent, and a gapped boundary read as a clean
`<|endoftext|><|endofconversation|>[USER] ...` join. `make sanity-sample
ARGS="--data data/warm_start.pt"` (`work:sanity`, log
`sft/logs/sanity-sample-20261011-134419.log`) EXIT=0; random windows read as
ordinary conversations and Dolly rows. Corpus accepted.

Training (chunk 512, batch 6, trainable 21,323,264, VRAM free 94.0 GB at
start): loss 1.83 (step 1) → 1.28 (step 4) → 2.51 (8) → 5.14 (9) → 10.15
(14) → 12.26 (20) with gnorm up to 3984 during LR warm-up (lr ≤ 6e-5), then
a slow fall: 7.4 at step 74, 6.28 at step 136. I stopped it at 13:48 (step
136) as a suspected numerical fault: AdamW at lr ≤ 3e-5 with clip 1.0 cannot
move a rank-16 LoRA that far in four steps, and a loss near ln(50k) means a
near-uniform model.

Checks while stopped: `tests/experiments/dreams/test_mixer_fused.py` on the
GH200 (`work:gputest`, log `sft/logs/gputest-mixer-fused.log`): 5 passed in
91 s. Library versions: python 3.14.8, torch 2.11.0+cu128, mamba_ssm
2.3.2.post1, triton 3.6.0 (the 08-10 warm start ran torch 2.13.0+cu129).

Teammate message at 13:50 (altrup, not watching live): this spike is the
known shape of the 2.7B warm start. They placed the 08-10 run's every-10th
step lines at `/tmp/warm-start-20260810-steps.txt` (not in the repo). Those
read: step 1 loss 2.18, step 21 12.12 (gnorm 692), step 31 10.52 (gnorm
5440), step 86 7.54, step 201 2.62, step 261 1.41. My run at step 86 read
7.36: the same curve. The 08-10 note's "5.76 → 4.77 over the first 160
steps" was that session's mis-summary. Decision: treat it as the registered
training's known transient, finish the fused test (done, passed), delete the
partial `models/mamba2_2_7b/checkpoints/epoch-1`, rerun `make warm-start`
unchanged. Open for discussion, not a stop: a transient collapse to
near-uniform during warm-up and a relearn from there is a different object
from a gently adapted base, even when step 800 passes acceptance; a warm-up
schedule or lower peak LR is a protocol change and belongs to the debrief.

## 13:52 UTC — warm start attempt 2 (verbatim, `train`, after the HF_TOKEN export)

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start; echo "EXIT=$?" >> "$(ls -t logs/warm-start-*.log | head -1)"
