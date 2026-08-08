# Which sft/data/ artifacts travel between this machine and the instance.
# Sourced (not executed) by lambda_launch.sh (upload) and lambda_pull.sh
# (pull-back) so both ends agree on one list: an artifact the box generates is
# pulled home, and uploaded again on the next launch instead of regenerated.
#
# Space-separated basename globs, matched inside sft/data/. The default takes
# the finished train_*/eval_* artifacts and their scored *-filtered variants
# (whose embedded per-item scores make any threshold policy a local
# filter_items.py --rescore away), plus the dream-sleep grid's shared inputs --
# the per-seed dream caches every arm distils, their decoded sidecars, and the
# self-calibrated knowledge battery, all of which must stay identical across
# runs and machines for a cell to be comparable. Leaves pilot_*/smoke_* scratch
# files and raw intermediates behind. Override with LAMBDA_DATA_ARTIFACTS in
# scripts/.env. Empty or unset means the defaults: to deliberately sync
# nothing, set a non-matching pattern (e.g. "none-*") — an empty string
# silently disabling the pull is how the g2 caches were lost.
DATA_ARTIFACTS="${LAMBDA_DATA_ARTIFACTS:-train*.pt eval_*.pt dream_cache_*.pt dream_*.txt knowledge_battery_*.json}"

_data_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Local matches, as rsync -R sources anchored at the repo root, so they land at
# the same sft/data/ path on the other machine. Also what the launch confirm
# screen lists — the listing and the transfer are the same array, so they
# cannot disagree.
data_local_files=()
for _pat in $DATA_ARTIFACTS; do
  for _f in "$_data_root"/sft/data/$_pat; do
    [[ -f "$_f" ]] && data_local_files+=("$_data_root/./sft/data/${_f##*/}")
  done
done

# The same selection expressed as an rsync filter, for the remote end where the
# globbing isn't ours to do. '*/' keeps the directories --relative implies.
data_filter=(--include='*/')
for _pat in $DATA_ARTIFACTS; do data_filter+=(--include="$_pat"); done
data_filter+=(--exclude='*')
unset _pat _f
