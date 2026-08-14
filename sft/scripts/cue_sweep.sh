#!/usr/bin/env bash
# Rehearsal-cue sweep: which (temperature, dream-prompt) makes the dream
# actually rehearse the wake facts? Dream-only diagnostic -- distill-steps 1,
# so the run cost is the dream, not the training.
set -u
i=0
for temp in 1.0 0.7 0.4; do
  for cue in "" "Let me go back over what I was just told." "Here is a summary of the codes I was given."; do
    i=$((i + 1))
    echo "=== cell $i: temp=$temp cue='$cue' ==="
    MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
      uv run --no-sync python -u -m experiments.dreams.cli \
        --arm replay --n-facts 4 --filler-tokens 40 --dream-tokens 512 \
        --distill-steps 1 --seed 1234 --dream-temp "$temp" --dream-prompt "$cue" \
        --out "logs/cue_sweep_$i.jsonl" 2>&1 | grep -E "rehearsal fraction|decoded dream" -A3
  done
done
echo "SWEEP_EXIT=0"
