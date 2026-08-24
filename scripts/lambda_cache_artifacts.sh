# Cache artifacts selected for symmetric upload and pull.
# Entries are space-separated paths or globs relative to the repo's .cache/.
CACHE_ARTIFACTS="${LAMBDA_CACHE_ARTIFACTS:-}"

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_cache_root="$_repo_root/.cache"
cache_local_paths=()
read -ra _cache_patterns <<< "$CACHE_ARTIFACTS"
for _pat in "${_cache_patterns[@]}"; do
  if [[ ! "$_pat" =~ ^[A-Za-z0-9._/*-]+$ ]] \
    || [[ "$_pat" == /* || "$_pat" == "." || "$_pat" == "*" \
    || "$_pat" == ".." || "$_pat" == ../* || "$_pat" == */../* || "$_pat" == */.. ]]; then
    echo "error: unsafe LAMBDA_CACHE_ARTIFACTS entry: $_pat" >&2
    return 1 2>/dev/null || exit 1
  fi
  for _f in "$_cache_root"/$_pat; do
    [[ -e "$_f" ]] && cache_local_paths+=("$_repo_root/./.cache/${_f#"$_cache_root"/}")
  done
done
unset _repo_root _cache_root _cache_patterns _pat _f
