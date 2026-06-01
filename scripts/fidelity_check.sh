#!/usr/bin/env bash
# Run the passthrough fidelity A/B/A check: replay one fixed trace direct-to-vLLM
# vs through-the-router and compare TTFT/TPOT/e2e. MODEL must match the served
# model id. Edit the values below to change the model, URLs, or trace size.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen2.5-0.5B-Instruct
DIRECT_URL=http://127.0.0.1:8001
ROUTER_URL=http://127.0.0.1:8000
N=60
RATE=20
MAX_TOKENS=64
ROUTER_LOG=logs/router_log.jsonl
PYTHON=python   # change to python3 / a venv path if your interpreter isn't `python`

exec "$PYTHON" tests/fidelity_check.py \
  --direct "$DIRECT_URL" \
  --router "$ROUTER_URL" \
  --model "$MODEL" \
  --make-trace --n "$N" --rate "$RATE" --max-tokens "$MAX_TOKENS" \
  --router-log "$ROUTER_LOG"
