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
