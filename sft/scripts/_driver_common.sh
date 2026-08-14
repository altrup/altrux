# Shared front matter for the sec 3.1 grid drivers.
stamp() { date +%H:%M:%S; }

# The registered protocol for this session is warm-start-everything (sec 3.1):
# a cold run is a different experiment, so it has to be asked for by name.
# Takes the caller's own usage example; sets init_adapter and init_flag.
require_init_adapter() {
  init_adapter=${INIT_ADAPTER:-}
  if [ -z "$init_adapter" ]; then
    echo "refusing to run: INIT_ADAPTER is unset, and sec 3.1 registers the warm start for the whole" >&2
    echo "session -- cache, battery and every arm. Pass the checkpoint dir (a train.py step-N/):" >&2
    echo "  $1" >&2
    echo "or INIT_ADAPTER=none for a deliberate cold run (the g2 reference shape)." >&2
    exit 2
  fi
  if [ "$init_adapter" = none ]; then
    init_flag=""
    echo "[$(stamp)] === COLD RUN: INIT_ADAPTER=none, no warm start -- not the sec 3.1 protocol ==="
  else
    init_flag="--init-adapter $init_adapter"
    [ -d "$init_adapter" ] || { echo "no warm-start checkpoint dir at $init_adapter" >&2; exit 2; }
  fi
}
