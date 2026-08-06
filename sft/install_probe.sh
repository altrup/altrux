#!/usr/bin/env bash
# Can ANY budget install a fact at all? The registered grid returned 0/4
# installs for every arm at 200 and 800 steps, and a frontier cannot be drawn
# through zeros. The null installed 4/9 at this same LR and step count, but its
# replay material was fact-bearing throughout while this dream is ~80% filler
# (measured rehearsal 0.21) -- so the knobs under test are rehearsal DENSITY
# and total rehearsal count, with one learning-rate cell to bound that too.
#
# Arm A only: it is the arm that moved most (dlogp +1.8 vs +0.08/-0.07), so it
# is the cheapest test of whether installation is reachable at all here.
set -u
cd ~/altrux/sft
probe() {
  local name=$1; shift
  echo "=== install $name ==="
  MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
    uv run --no-sync python -u dream_sleep.py --arm replay --n-facts 4 --filler-tokens 40 \
      --dream-temp 0.7 --cue-greedy 12 --seed 1234 "$@" 2>&1 \
    | grep --line-buffered -E "rehearsal fraction|probe w1|battery retained|held-out ppl 2|dPPL"
}
probe dense32_d800   --cue-every 32 --dream-tokens 512  --distill-steps 800  --out logs/inst_a.jsonl
probe long2048_d1600 --cue-every 64 --dream-tokens 2048 --distill-steps 1600 --out logs/inst_b.jsonl
probe dense32_lr3e4  --cue-every 32 --dream-tokens 1024 --distill-steps 1600 --lr 3e-4 --out logs/inst_c.jsonl
echo INSTALL_PROBE_DONE
