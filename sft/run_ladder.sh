#!/usr/bin/env bash
# The saturation ladder (DISCUSSION-20260806 sec 5b), seed 1234, one rung per
# invocation: run_ladder.sh 3200 [arm...]
#
# Cells are named <arm>_d<steps> so summarize_grid.py's default glob picks them
# up and pools each rung as its own arm, alongside seed 1234's no-sleep cell --
# which the floor correction needs, since a raw margin has no zero (1f1ab33).
#
# B1 is on the ladder although sec 5b registered only A and B2: it was the best
# gentle learner of the single-sleep grid, at 1/348th of A's token-gradients.
# The registered stop rule still applies per arm -- a rung flat against d800
# ends that arm's ladder rather than buying the next rung.
set -u
steps=${1:?usage: run_ladder.sh STEPS [arm...]}
shift
arms=("$@")
[ ${#arms[@]} -eq 0 ] && arms=(A B1 B2 B2deep)
seed=1234
cd ~/altrux/sft
env_vars="MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.."
common="--n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every 32 --cue-greedy 12"
stamp() { date +%H:%M:%S; }

arm_flags() {
  case $1 in
    A)      echo "--arm replay" ;;
    B1)     echo "--arm drain" ;;
    B2)     echo "--arm counterfactual" ;;
    B2deep) echo "--arm counterfactual --deep" ;;
    *)      echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for arm in "${arms[@]}"; do
  out="logs/g2_${arm}_d${steps}_s${seed}.jsonl"
  if grep -q '"phase": "done"' "$out" 2>/dev/null; then
    echo "[$(stamp)] === skip ${arm}_d${steps} (done) ==="; continue
  fi
  echo "[$(stamp)] === ladder ${arm}_d${steps} seed=$seed ==="
  env $env_vars uv run --no-sync python -u dream_sleep.py $common --seed "$seed" \
      $(arm_flags "$arm") --lr 1e-4 --distill-steps "$steps" --probe-every 1600 \
      --out "$out" 2>&1 | tee "logs/g2_${arm}_d${steps}_s${seed}.log"
done
echo "[$(stamp)] LADDER_EXIT_${steps}=0"
