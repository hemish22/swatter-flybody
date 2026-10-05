#!/usr/bin/env bash
# Run SWATTER's heavy offline jobs on the DGX instead of your laptop.
#
# The rule this enforces: the expensive steps (the 1.05 GB connectome table, the
# LIF sweeps, the ablations) belong on the DGX per the compute plan. Local runs
# stay for cheap things: stimuli, lint, tests, reading a manifest.
#
# Usage:
#   ./scripts/dgx.sh setup                  one-time: push code, make a venv
#   ./scripts/dgx.sh fetch                  download the MaleCNS tables (~1.1 GB)
#   ./scripts/dgx.sh extract                build the escape subgraph
#   ./scripts/dgx.sh run <job> [args...]    any offline/ module, e.g. loom_sweep
#   ./scripts/dgx.sh status                 is anything running over there
#   ./scripts/dgx.sh logs <job>             tail the last run's log
#   ./scripts/dgx.sh pull [path]            fetch results back (default: data/subgraph)
#   ./scripts/dgx.sh shell                  interactive shell on the DGX
#
# Jobs are module names under offline/, e.g. `fetch_malecns`, `extract_subgraph`,
# `loom_sweep`. Anything passed after the job name is forwarded verbatim, so
#   ./scripts/dgx.sh run extract_subgraph --min-weight 5
# works without this script knowing anything about the flag.
#
# Host: SWATTER_DGX_HOST (default: h4hgpu). Remote dir: SWATTER_REMOTE_DIR
# (default: ~/swatter). Sync is one-way (laptop -> DGX) for code, and results
# come back with `pull`, so the two machines never fight over a file.

set -euo pipefail

HOST="${SWATTER_DGX_HOST:-h4hgpu}"
REMOTE_DIR="${SWATTER_REMOTE_DIR:-~/swatter-flybody}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_PREFIX="swatter"

# Code and results. data/raw is deliberately absent: 1.1 GB of connectome
# tables are downloaded once on the DGX and never copied back down.
SYNC_INCLUDE=(
  "offline/"
  "web/"
  "engine/"
  "server/"
  "docs/"
  "Makefile"
  "README.md"
  "AGENTS.md"
  "pyproject.toml"
)
EXCLUDE=(
  "--exclude=.venv"
  "--exclude=data/raw"
  "--exclude=.git"
  "--exclude=node_modules"
  "--exclude=__pycache__"
  "--exclude=.DS_Store"
  "--exclude=data/subgraph"
  "--exclude=docs/figures"
)

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

require_host() {
  if [[ "$HOST" == "h4hgpu" || "$HOST" == *"172.16."* ]]; then
    log "host $HOST looks like a tailnet/VPN address; if the connection times"
    log "out, bring the VPN up first, then retry."
  fi
}

sync_code() {
  log "syncing code -> $HOST:$REMOTE_DIR"
  ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
  rsync -az --delete "${EXCLUDE[@]}" "${SYNC_INCLUDE[@]/#/$REPO_ROOT/}" "$HOST:$REMOTE_DIR/"
}

remote_python() {
  # SWATTER_REMOTE=1 is what satisfies the local-execution guard in
  # offline/common.py. SWATTER_DATA keeps the big tables under the remote data dir.
  ssh "$HOST" "cd '$REMOTE_DIR' && SWATTER_REMOTE=1 SWATTER_DATA=data '$@'"
}

cmd_setup() {
  require_host
  sync_code
  log "creating the venv and installing deps on $HOST"
  ssh "$HOST" "cd '$REMOTE_DIR' && (command -v uv >/dev/null && uv sync --frozen || (python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt))"
  log "checking the DGX is actually a DGX"
  ssh "$HOST" "nvidia-smi -L 2>/dev/null | head -4 || echo '(no nvidia-smi: this host has no GPU, sweeps will be CPU-only)'"
  log "setup done. Next: ./scripts/dgx.sh run fetch_malecns"
}

cmd_run() {
  local job="${1:?usage: dgx.sh run <job> [args...]}"
  shift
  require_host
  [[ -f "$REPO_ROOT/offline/$job.py" ]] || die "no offline/$job.py; jobs are modules under offline/"

  sync_code
  local session="${SESSION_PREFIX}-$(date +%H%M%S)"

  log "launching $session: offline/$job.py $*"
  # Detached tmux so a dropped VPN does not kill a long sweep.
  ssh -t "$HOST" "tmux new-session -d -s '$session' \
    'cd \"$REMOTE_DIR\" && SWATTER_REMOTE=1 SWATTER_DATA=data .venv/bin/python offline/$job.py $* \
       > logs/$job.log 2>&1; echo \"exit=\$?\" >> logs/$job.log; sleep 3600'"

  log "running detached as tmux session '$session'"
  echo
  echo "  follow:   ./scripts/dgx.sh logs $job"
  echo "  status:   ./scripts/dgx.sh status"
  echo "  results:  ./scripts/dgx.sh pull"
  echo
}

cmd_status() {
  ssh "$HOST" "tmux ls 2>/dev/null | grep '^$SESSION_PREFIX' || echo 'no swatter sessions running'"
  echo
  ssh "$HOST" "ls -la '$REMOTE_DIR'/logs/ 2>/dev/null | tail -8 || echo 'no logs yet'"
}

cmd_logs() {
  local job="${1:?usage: dgx.sh logs <job>}"
  ssh -t "$HOST" "touch '$REMOTE_DIR/logs/$job.log' && tail -f '$REMOTE_DIR/logs/$job.log'"
}

cmd_pull() {
  local path="${1:-data/subgraph}"
  log "pulling $path from $HOST"
  mkdir -p "$REPO_ROOT/$(dirname "$path")"
  rsync -az "$HOST:$REMOTE_DIR/$path" "$REPO_ROOT/$(dirname "$path")/"
  log "got $path -> $REPO_ROOT/$(dirname "$path")"
}

cmd_shell() {
  ssh -t "$HOST" "cd '$REMOTE_DIR' && SWATTER_REMOTE=1 SWATTER_DATA=data bash"
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  run) shift; cmd_run "$@" ;;
  # Aliases for the two setup steps, so the common path needs no job names.
  fetch) shift; cmd_run fetch_malecns --group core --columns "$@" ;;
  extract) shift; cmd_run extract_subgraph "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  pull) shift; cmd_pull "$@" ;;
  shell) shift; cmd_shell "$@" ;;
  ""|-h|--help|help) sed -n '2,30p' "$0" ;;
  *) die "unknown command '$1'; try: setup, fetch, extract, run, status, logs, pull, shell" ;;
esac
