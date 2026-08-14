#!/usr/bin/env bash
# Sweep 3: is temp-0.7 coverage seed-robust, and does dream length actually buy
# coverage? Sweeps 1-2 disagreed at n=1 (512 and 2048 both scored 3/4 entities,
# 1024 scored 2/4), so decide across 3 seeds rather than tune on one sample.
# Coverage is the gating variable for the whole A/B: a fact the dream never
# rehearses cannot be installed by any arm.
set -u
for seed in 1234 2345 3456; do
  for toks in 512 2048; do
    echo "=== cell3 seed=$seed tokens=$toks ==="
    MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
      uv run --no-sync python -u -m experiments.dreams.cli \
        --arm replay --n-facts 4 --filler-tokens 40 --dream-tokens "$toks" \
        --distill-steps 1 --seed "$seed" --dream-temp 0.7 \
        --out "logs/seed_sweep_${seed}_${toks}.jsonl" 2>&1 | grep -E "rehearsal fraction"
  done
done
echo "SWEEP3_EXIT=0"
