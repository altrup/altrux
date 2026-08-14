#!/usr/bin/env bash
# The single-sleep primary of the CL A/B (DISCUSSION-20260805 sec 4, box docket
# sec 5.2), one seed per invocation so seeds run as parallel streams -- the A10
# sits at ~2.3/23 GiB and ~20% util on one run, so the grid is latency-bound,
# not memory-bound.
#
# Deviations from the docket's registered invocation, both forced by this
# session's rehearsal sweeps and recorded in the run notes:
#   --dream-temp 0.7   : at the default 1.0 the dream never rehearses a fact
#                        (rehearsal 0.00), which makes every arm untestable.
#                        Below ~0.7 it mode-collapses onto one fact.
#   --cue-every 64 --cue-greedy 12 : free dreams rehearse by luck (code
#                        coverage 4/4, 0/4, 0/4 across seeds). Cued: 4/4
#                        entities everywhere, 3-4/4 codes.
# Everything else is the registered configuration.
set -u
seed=${1:?usage: make grid ARGS=SEED}
common="--n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every 64 --cue-greedy 12"

run() {  # run <name> <args...>
  local name=$1; shift
  local out="logs/dream_${name}_s${seed}.jsonl"
  # A cell is finished only once it has written its locality record: the jsonl
  # is non-empty from the first probe, so existence alone would skip a cell a
  # kill interrupted halfway.
  if grep -q '"phase": "locality"' "$out" 2>/dev/null; then echo "=== skip $name s$seed (done) ==="; return; fi
  echo "=== grid $name seed=$seed ==="
  MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
    uv run --no-sync python -u -m experiments.dreams.cli $common --seed "$seed" "$@" --out "$out" 2>&1 \
    | grep --line-buffered -E "rehearsal fraction|probe w|battery retained|held-out ppl|in_context w|WARNING|Error"
}

for steps in 200 800; do
  for arm in replay drain counterfactual; do
    run "${arm}_d${steps}" --arm "$arm" --distill-steps "$steps"
  done
done
run sftref_d200 --sft-ref --distill-steps 200
run sftref_d800 --sft-ref --distill-steps 800
run nosleep     --no-sleep
[ "$seed" = 1234 ] && run drainlive --arm drain-live

echo "GRID_EXIT_${seed}=0"
