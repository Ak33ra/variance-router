#!/usr/bin/env bash
# Launch N independent vLLM instances, ONE PER GPU — the real multi-GPU topology
# (one `vllm serve` per GPU, distinct port, each pinned via CUDA_VISIBLE_DEVICES,
# NO --data-parallel-size) — AND write the matching router config (cluster.yaml)
# so the backend list always agrees with what's actually serving. Ctrl-C stops all.
#
# Edit GPUS to match your machine. Ports are BASE_PORT, BASE_PORT+1, ... in GPU
# order (GPUS[i] -> port BASE_PORT+i). The router policy lives in the written
# config too — edit POLICY/POLICY_PARAMS below per run (jsq baseline, burst, ...).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen3.5-9B
GPUS=(0 1)        # one vLLM instance per listed GPU id
BASE_PORT=8001        # instance i serves on BASE_PORT + i
GPU_MEM_UTIL=0.90     # each instance owns its GPU, so use most of it
MAX_MODEL_LEN=2048

# --- router config written for this cluster ---
CONFIG=cluster.yaml
BACKEND_HOST=127.0.0.1
ROUTER_HOST=127.0.0.1
ROUTER_PORT=8000
POLICY=burst_wrap            # baseline; set to burst / regime_aware for the interventions
POLICY_PARAMS="{base: jsq, burst_size 8}"    # e.g. {burst_size: 8} for burst
#POLICY_PARAMS="{}"
LOG_PATH=logs/router_burstjsq_log.jsonl

command -v vllm >/dev/null || { echo "vllm not found on PATH (install it separately)"; exit 1; }

# Each vLLM instance must own a SEPARATE GPU (no data-parallel, no GPU sharing).
uniq_gpus=$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)
[ "$uniq_gpus" -eq "${#GPUS[@]}" ] || {
  echo "GPUS has duplicate ids (${GPUS[*]}); each instance must be on its own GPU"; exit 1; }

# Write the router config. gpu_id is the PHYSICAL CUDA device (GPUS[i]), not the
# backend index, so per-node logs attribute to the right GPU even if GPUS is
# remapped (e.g. (2 3)).
{
  echo "backends:"
  for i in "${!GPUS[@]}"; do
    echo "  - {host: ${BACKEND_HOST}, port: $((BASE_PORT + i)), gpu_id: ${GPUS[$i]}}"
  done
  echo "policy: {name: ${POLICY}, params: ${POLICY_PARAMS}}"
  echo "log_path: ${LOG_PATH}"
  echo "host: ${ROUTER_HOST}"
  echo "port: ${ROUTER_PORT}"
} > "$CONFIG"
echo "wrote ${CONFIG}:"; sed 's/^/  /' "$CONFIG"
echo

# Launch one instance per GPU.
pids=()
cleanup() { [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  port=$((BASE_PORT + i))
  echo "GPU ${gpu} -> ${MODEL} on :${port}"
  CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$MODEL" --port "$port" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" &
  pids+=($!)
done

echo
echo "started ${#pids[@]} instances. In another terminal start the router with:"
echo "  ./scripts/start_router.sh ${CONFIG}"
echo
wait
