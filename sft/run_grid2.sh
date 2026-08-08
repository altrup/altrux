#!/usr/bin/env bash
# The d800 baselines of DISCUSSION-20260807 sec 4(3), one seed per invocation:
#
#   INIT_ADAPTER=../models/mamba2_780m/checkpoints/warm_start.pt ERASE_OP=deflated ./run_grid2.sh 1234
#
# Concurrency is a per-card decision, not a property of this script (sft/CLAUDE.md):
# serial on an A10, three seeds at once on a GH200.
#
# Result files use the g2_ prefix so the corrected B1 cannot collide with the
# retired drain arm's 08-06 filenames in the summarizer's glob or in this
# script's resume check. The erase-family cells carry the operator in their
# name, so the sec 4(2) picker can run raw and deflated at the same seed
# without one overwriting the other; the no-sleep floor cell must stay named
# exactly `nosleep` -- summarize_grid.apply_floor keys the per-fact floor every
# Delta metric is computed against on that arm name.
#
# Every arm distils the ONE dream this seed's cache holds; the cache is built
# first, by this script, from the same warm-started weights the arms run at
# (a cache whose generator disagrees with the cell's --init-adapter is fatal),
# and every cell records its hashes so summarize_grid.py can assert they agree.
set -u
seed=${1:?usage: INIT_ADAPTER=<ckpt|none> [ERASE_OP=deflated] run_grid2.sh SEED}
# A seed whose dream binds fewer than 3/4 codes to their own entity is rebuilt
# at CUE_EVERY=24 (DISCUSSION-20260806 sec 5a); the arms of that seed run at the
# same value so the cell's flags regenerate the cache they distilled.
cue_every=${CUE_EVERY:-32}
erase_op=${ERASE_OP:-deflated}
stamp() { date +%H:%M:%S; }

# The registered protocol for this session is warm-start-everything (sec 3.1):
# a cold grid is a different experiment, so it has to be asked for by name.
init_adapter=${INIT_ADAPTER:-}
if [ -z "$init_adapter" ]; then
  echo "refusing to run: INIT_ADAPTER is unset, and sec 3.1 registers the warm start for the whole" >&2
  echo "session -- cache, battery and every arm. Pass the checkpoint:" >&2
  echo "  INIT_ADAPTER=../models/mamba2_780m/checkpoints/warm_start.pt ./run_grid2.sh $seed" >&2
  echo "or INIT_ADAPTER=none for a deliberate cold run (the g2 reference shape)." >&2
  exit 2
fi
if [ "$init_adapter" = none ]; then
  init_flag=""
  echo "[$(stamp)] === COLD RUN: INIT_ADAPTER=none, no warm start -- not the sec 3.1 protocol ==="
else
  init_flag="--init-adapter $init_adapter"
  [ -f "$init_adapter" ] || { echo "no warm-start checkpoint at $init_adapter" >&2; exit 2; }
fi
echo "[$(stamp)] === seed $seed | erase op $erase_op | warm start ${init_adapter} | cue_every $cue_every ==="

cd ~/altrux/sft
env_vars="MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.."
common="--n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every $cue_every --cue-greedy 12"
regime="--lr 1e-4 --distill-steps 800 --probe-every 200 --erase-op $erase_op"

cache="data/dream_cache_s${seed}.pt"
if [ ! -f "$cache" ]; then
  echo "[$(stamp)] === building dream cache: seed=$seed ==="
  env $env_vars uv run --no-sync python -u dream_sleep.py $common $init_flag --seed "$seed" \
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
  # A cell is finished only once it has written its done record. Locality
  # records are written by every periodic probe, so keying on those would skip
  # a cell that died -- or is still running -- after its first probe point.
  if grep -q '"phase": "done"' "$out" 2>/dev/null; then
    echo "[$(stamp)] === skip $name s$seed (done) ==="; return
  fi
  echo "[$(stamp)] === grid $name seed=$seed erase=$erase_op ==="
  env $env_vars uv run --no-sync python -u dream_sleep.py $common $regime $init_flag --seed "$seed" "$@" \
      --out "$out" 2>&1 \
    | tee "logs/g2_${name}_s${seed}.log" \
    | grep --line-buffered -E "rehearsal fraction|bound|probe w|battery retained|held-out ppl|in_context w|token-gradient|dream_sha|equivalence|WARNING|Error"
}

run A                    --arm replay              # full-sequence chunking, fresh state
run "B1_$erase_op"       --arm drain               # ablate in place, carry the student's state
run "B2fd_$erase_op"     --arm b2-fused-detached   # one step per pass, spine detached
run "B2fdeep_$erase_op"  --arm b2-fused-deep       # the same, BPTT through the scan
run "B3f_$erase_op"      --arm b3-fused            # the same, on the generator's own spine
run "B2tok_$erase_op"    --arm counterfactual      # per-token B2: the g2 bridge cell
# The floor every Delta metric is read against. Trains on nothing, so it is
# erase-operator-independent and one cell per seed serves every picker arm.
run nosleep              --no-sleep

echo "[$(stamp)] GRID2_EXIT_${seed}=0"
