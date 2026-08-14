#!/usr/bin/env bash
# Sweep 2: dream length x temperature, no cue (sweep 1 showed cues do not help
# and temperature is the real lever). Scored on distinct-fact COVERAGE, not
# rehearsal fraction -- sweep 1 cell 7 hit the highest fraction (0.312) by
# looping one fact 35 times, which can only ever install that one fact.
set -u
i=0
for temp in 0.7 0.55; do
  for toks in 1024 2048; do
    i=$((i + 1))
    echo "=== cell2 $i: temp=$temp dream_tokens=$toks ==="
    MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
      uv run --no-sync python -u -m experiments.dreams.cli \
        --arm replay --n-facts 4 --filler-tokens 40 --dream-tokens "$toks" \
        --distill-steps 1 --seed 1234 --dream-temp "$temp" \
        --out "logs/len_sweep_$i.jsonl" 2>&1 | grep -E "rehearsal fraction"
  done
done
echo "SWEEP2_EXIT=0"
