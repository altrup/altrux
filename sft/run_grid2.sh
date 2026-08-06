#!/usr/bin/env bash
# The corrected single-sleep grid (DISCUSSION-20260806 sec 5a), one seed per
# invocation. SERIAL by construction: three concurrent streams measured ~4x
# worse aggregate throughput than one on the A10, and these loops are
# latency-bound, not memory-bound.
#
# Result files use the g2_ prefix so the corrected B1 cannot collide with the
# retired drain arm's 08-06 filenames in the summarizer's glob or in this
# script's resume check.
#
# Every arm distils the ONE dream this seed's cache holds; the cache is built
# first, by this script, and every cell records its hashes so
# summarize_grid.py can assert they agree.
set -u
seed=${1:?usage: run_grid2.sh SEED}
# A seed whose dream binds fewer than 3/4 codes to their own entity is rebuilt
# at CUE_EVERY=24 (DISCUSSION-20260806 sec 5a); the arms of that seed run at the
# same value so the cell's flags regenerate the cache they distilled.
cue_every=${CUE_EVERY:-32}
cd ~/altrux/sft
env_vars="MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.."
common="--n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every $cue_every --cue-greedy 12"
regime="--lr 1e-4 --distill-steps 800 --probe-every 200"
stamp() { date +%H:%M:%S; }

cache="data/dream_cache_s${seed}.pt"
if [ ! -f "$cache" ]; then
  echo "[$(stamp)] === building dream cache: seed=$seed ==="
  env $env_vars uv run --no-sync python -u dream_sleep.py $common --seed "$seed" \
    --build-dream-cache 2>&1 | tee "logs/g2_cache_s${seed}.log"
  # Read the sidecar before running a single arm: a seed whose binding-aware
  # coverage is below 3/4 gets its cache regenerated at --cue-every 24 first.
  echo "[$(stamp)] cache built -- read data/dream_s${seed}.txt before continuing"
else
  echo "[$(stamp)] === dream cache $cache present; no arm regenerates ==="
fi

run() {  # run <name> <args...>
  local name=$1; shift
  local out="logs/g2_${name}_s${seed}.jsonl"
  # A cell is finished only once it has written its locality record: the jsonl
  # is non-empty from the first probe, so existence alone would skip a cell a
  # kill interrupted halfway.
  if grep -q '"phase": "locality"' "$out" 2>/dev/null; then
    echo "[$(stamp)] === skip $name s$seed (done) ==="; return
  fi
  echo "[$(stamp)] === grid $name seed=$seed ==="
  env $env_vars uv run --no-sync python -u dream_sleep.py $common $regime --seed "$seed" "$@" \
      --out "$out" 2>&1 \
    | tee "logs/g2_${name}_s${seed}.log" \
    | grep --line-buffered -E "rehearsal fraction|bound|probe w|battery retained|held-out ppl|in_context w|token-gradient|dream_sha|WARNING|Error"
}

run A       --arm replay                     # full-sequence chunking, fresh state
run B1      --arm drain                      # corrected: ablate in place, carry the student's state
run B2      --arm counterfactual             # ablate a copy, carry the intact state
run CE      --arm replay --ce-on-dream
run sftref  --sft-ref
run nosleep --no-sleep
# Bridge cell: the 08-06 configuration (chunk 48, carried within pass), so the
# old numbers stay interpretable. Seed 1234 only.
[ "$seed" = 1234 ] && run A_bridge --arm replay --chunk-len 48

echo "[$(stamp)] GRID2_EXIT_${seed}=0"
