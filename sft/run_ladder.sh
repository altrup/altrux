#!/usr/bin/env bash
# The saturation ladder (DISCUSSION-20260807 sec 4(4)), seed 1234, one rung per
# invocation:
#
#   INIT_ADAPTER=../models/mamba2_780m/checkpoints/warm_start.pt ERASE_OP=deflated ./run_ladder.sh 3200 [arm...]
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
steps=${1:?usage: INIT_ADAPTER=<ckpt|none> [ERASE_OP=deflated] run_ladder.sh STEPS [arm...]}
shift
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(A B1 B2fd B2fdeep)
seed=1234
erase_op=${ERASE_OP:-deflated}
stamp() { date +%H:%M:%S; }

# The registered protocol for this session is warm-start-everything (sec 3.1),
# ladder included; a cold rung is a different experiment and has to be asked
# for by name.
init_adapter=${INIT_ADAPTER:-}
if [ -z "$init_adapter" ]; then
  echo "refusing to run: INIT_ADAPTER is unset, and sec 3.1 registers the warm start for the whole" >&2
  echo "session, ladder included. Pass the checkpoint:" >&2
  echo "  INIT_ADAPTER=../models/mamba2_780m/checkpoints/warm_start.pt ./run_ladder.sh $steps" >&2
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
echo "[$(stamp)] === ladder d$steps | erase op $erase_op | warm start ${init_adapter} | arms ${arms[*]} ==="

cd ~/altrux/sft
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
  env $env_vars uv run --no-sync python -u dream_sleep.py $common $init_flag --seed "$seed" \
      $(arm_flags "$arm") --lr 1e-4 --distill-steps "$steps" --probe-every 1600 \
      --erase-op "$erase_op" --out "$out" 2>&1 | tee "logs/g2_${arm}_d${steps}_s${seed}.log"
done
echo "[$(stamp)] LADDER_EXIT_${steps}=0"
