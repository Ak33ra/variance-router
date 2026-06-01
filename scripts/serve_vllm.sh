#!/usr/bin/env bash
# Start ONE independent vLLM OpenAI server (one GPU, one port) for the router to
# front. Runs in the foreground; Ctrl-C stops it. Override defaults via env, e.g.
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8001 ./scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8001}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

command -v vllm >/dev/null || { echo "vllm not found on PATH (install it separately)"; exit 1; }

echo "serving ${MODEL} on :${PORT} (gpu-mem-util=${GPU_MEM_UTIL}, max-model-len=${MAX_MODEL_LEN})"
exec vllm serve "$MODEL" --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN"
