#!/usr/bin/env bash
# Sweep 4: retrieval-stem seeding, tested on the seeds where free generation
# fails (2345: zero rehearsal at both lengths; 3456: 1/4 entities, 0/4 codes).
#
# Sweep 1's cues were declarative ("let me review...") and cued a summarizing
# register the dream then drifted out of. This cue instead seeds the wake
# session's QUESTION stem and stops mid-phrase, so the only way to continue is
# to complete the entity from the wake state and then answer it -- i.e. the
# cue supplies the retrieval format and the state must supply the content.
#
# Deliberately entity-free and code-free: it cannot leak the scored answer, and
# being identical across arms it stays arm-neutral (sec 3 requires A and B2 to
# share a byte-identical dream at a seed).
set -u
cd ~/altrux/sft
STEM='[USER] What is the code for the'
for seed in 2345 3456 1234; do
  for cue in "$STEM" "Let me go back over the codes from earlier.$STEM"; do
    echo "=== cell4 seed=$seed cue='$cue' ==="
    MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
      uv run --no-sync python -u dream_sleep.py \
        --arm replay --n-facts 4 --filler-tokens 40 --dream-tokens 2048 \
        --distill-steps 1 --seed "$seed" --dream-temp 0.7 --dream-prompt "$cue" \
        --out "logs/stem_sweep_${seed}_${#cue}.jsonl" 2>&1 | grep -E "rehearsal fraction"
  done
done
echo "SWEEP4_EXIT=0"
