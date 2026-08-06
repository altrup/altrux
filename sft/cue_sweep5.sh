#!/usr/bin/env bash
# Sweep 5: does --cue-every give seed-robust coverage? This is the gate on the
# whole A/B -- sweeps 1-4 showed free generation rehearses usably at 1 of 3
# seeds and that no fixed cue string fixes it (cueing rescues seed 2345 and
# destroys seed 1234). Coverage must be ~4/4 codes at ALL THREE seeds for the
# grid to be worth running.
set -u
cd ~/altrux/sft
for seed in 1234 2345 3456; do
  for every in 64 128; do
    echo "=== cell5 seed=$seed cue_every=$every ==="
    MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
      uv run --no-sync python -u dream_sleep.py \
        --arm replay --n-facts 4 --filler-tokens 40 --dream-tokens 512 \
        --distill-steps 1 --seed "$seed" --dream-temp 0.7 --cue-every "$every" --cue-greedy 12 \
        --out "logs/cuedg_${seed}_${every}.jsonl" 2>&1 | grep -E "rehearsal fraction"
  done
done
echo "SWEEP5_EXIT=0"
