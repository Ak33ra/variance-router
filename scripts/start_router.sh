#!/usr/bin/env bash
# Start the router against a config file — the single entry point for launching
# the router on whatever config you give it. Pass the config path as the first
# argument; defaults to cluster.yaml (written by scripts/serve_cluster.sh).
#   ./scripts/start_router.sh cluster.yaml        # the real multi-GPU cluster
#   ./scripts/start_router.sh smoke_single.yaml   # the single-backend smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-cluster.yaml}"
PYTHON=python   # change to python3 / a venv path if your interpreter isn't `python`

[ -f "$CONFIG" ] || { echo "config not found: ${CONFIG} (run a serve_* script first, or pass a path)"; exit 1; }
echo "starting router on config ${CONFIG} (health-checks backends first)"
exec "$PYTHON" router.py --config "$CONFIG"
