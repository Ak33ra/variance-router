#!/usr/bin/env bash
# Run the passthrough fidelity A/B/A check: replay one fixed trace direct-to-vLLM
# vs through-the-router and compare TTFT/TPOT/e2e. MODEL must match the served
# model id. Override via env, e.g.
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct N=60 RATE=20 ./scripts/fidelity_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DIRECT_URL="${DIRECT_URL:-http://127.0.0.1:8001}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8000}"
N="${N:-60}"
RATE="${RATE:-20}"
MAX_TOKENS="${MAX_TOKENS:-64}"
ROUTER_LOG="${ROUTER_LOG:-logs/router_log.jsonl}"
PYTHON="${PYTHON:-python}"   # override if your interpreter isn't `python`, e.g. PYTHON=python3

exec "$PYTHON" tests/fidelity_check.py \
  --direct "$DIRECT_URL" \
  --router "$ROUTER_URL" \
  --model "$MODEL" \
  --make-trace --n "$N" --rate "$RATE" --max-tokens "$MAX_TOKENS" \
  --router-log "$ROUTER_LOG"
