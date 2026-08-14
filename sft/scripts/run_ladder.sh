#!/usr/bin/env bash
# The saturation ladder (DISCUSSION-20260807 sec 4(4)), seed 1234, one rung per
# invocation:
#
#   INIT_ADAPTER=../models/mamba2_780m/checkpoints/epoch-1/step-400 ERASE_OP=deflated make ladder ARGS="3200 [arm...]"
#
# Cells are named <arm>_d<steps> so summarize_grid.py's default glob picks them
# up and pools each rung as its own arm, alongside seed 1234's no-sleep cell --
# which the floor correction needs, since a raw margin has no zero (1f1ab33).
# run_grid2.sh emits that cell; the ladder does not.
#
# Default rungs are sec 4(4)'s list: A, B1, B2-fused-detached, B2-fused-deep.
# B3-fused earns rungs only if its d800 Delta-margin is within 2x of
# B2-fused-detached's, so it is nameable but never default. The registered stop
# rule applies per arm -- a rung flat against d800 ends that arm's ladder
# rather than buying the next rung.
set -u
source "$(dirname "$0")/_driver_common.sh"
steps=${1:?usage: INIT_ADAPTER=<ckpt|none> [ERASE_OP=deflated] make ladder ARGS="STEPS [arm...]"}
shift
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(A B1 B2fd B2fdeep)
seed=1234
erase_op=${ERASE_OP:-deflated}

require_init_adapter \
  "INIT_ADAPTER=../models/mamba2_780m/checkpoints/epoch-1/step-400 make ladder ARGS=$steps"
echo "[$(stamp)] === ladder d$steps | erase op $erase_op | warm start ${init_adapter} | arms ${arms[*]} ==="

env_vars="MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.."
common="--n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every 32 --cue-greedy 12"

arm_flags() {
  case $1 in
    A)                            echo "--arm replay" ;;
    B1)                           echo "--arm drain" ;;
    B2)                           echo "--arm counterfactual" ;;
    B2deep)                       echo "--arm counterfactual --deep" ;;
    B2fd|b2-fused-detached)       echo "--arm b2-fused-detached" ;;
    B2fdeep|b2-fused-deep)        echo "--arm b2-fused-deep" ;;
    B3f|b3-fused)                 echo "--arm b3-fused" ;;
    *)      echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for arm in "${arms[@]}"; do
  out="logs/g2_${arm}_d${steps}_s${seed}.jsonl"
  if grep -q '"phase": "done"' "$out" 2>/dev/null; then
    echo "[$(stamp)] === skip ${arm}_d${steps} (done) ==="; continue
  fi
  echo "[$(stamp)] === ladder ${arm}_d${steps} seed=$seed erase=$erase_op ==="
  env $env_vars uv run --no-sync python -u -m experiments.dreams.cli $common $init_flag --seed "$seed" \
      $(arm_flags "$arm") --lr 1e-4 --distill-steps "$steps" --probe-every 1600 \
      --erase-op "$erase_op" --out "$out" 2>&1 | tee "logs/g2_${arm}_d${steps}_s${seed}.log"
done
echo "[$(stamp)] LADDER_EXIT_${steps}=0"
