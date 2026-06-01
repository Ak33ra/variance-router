#!/usr/bin/env bash
# Generate a router config and launch the router. Defaults to a single backend
# (the passthrough smoke test, N=1). For the routing-mechanics test, edit
# BACKEND_PORTS to "8001 8002" and set POLICY=burst, POLICY_PARAMS='{burst_size: 8}'.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND_HOST=127.0.0.1
BACKEND_PORTS="8001 8002"   # space-separated; one backend per port
ROUTER_PORT=8000
POLICY=round_robin
POLICY_PARAMS="{}"
LOG_PATH=logs/router_log.jsonl
CONFIG=smoke_single.yaml
PYTHON=python   # change to python3 / a venv path if your interpreter isn't `python`

# Render the config (regenerated each run so the script fully defines the setup).
{
  echo "backends:"
  gpu=0
  for p in $BACKEND_PORTS; do
    echo "  - {host: ${BACKEND_HOST}, port: ${p}, gpu_id: ${gpu}}"
    gpu=$((gpu + 1))
  done
  echo "policy: {name: ${POLICY}, params: ${POLICY_PARAMS}}"
  echo "log_path: ${LOG_PATH}"
  echo "host: 127.0.0.1"
  echo "port: ${ROUTER_PORT}"
} > "$CONFIG"

echo "wrote ${CONFIG}:"; sed 's/^/  /' "$CONFIG"
echo "starting router on :${ROUTER_PORT} (health-checks backends first)"
exec "$PYTHON" router.py --config "$CONFIG"
