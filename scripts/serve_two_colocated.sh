#!/usr/bin/env bash
# Launch TWO vLLM instances co-located on one GPU (ports 8001/8002) to smoke-test
# routing mechanics across real backends. NOT a valid measurement — sharing a GPU
# confounds throughput; use only to confirm wiring/routing. Ctrl-C stops both.
# Edit the values below to change the model or per-instance memory settings.
set -euo pipefail

MODEL=Qwen/Qwen3.5-9B
GPU_MEM_UTIL=0.45
MAX_MODEL_LEN=1024
export CUDA_VISIBLE_DEVICES=0

command -v vllm >/dev/null || { echo "vllm not found on PATH (install it separately)"; exit 1; }

echo "serving two ${MODEL} instances on :8001 and :8002 (GPU ${CUDA_VISIBLE_DEVICES})"
vllm serve "$MODEL" --port 8001 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" &
PID1=$!
vllm serve "$MODEL" --port 8002 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" &
PID2=$!

trap 'kill "$PID1" "$PID2" 2>/dev/null || true' INT TERM EXIT
wait
