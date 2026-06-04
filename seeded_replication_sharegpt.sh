#!/bin/bash
set -euo pipefail

model="Qwen/Qwen3.5-9B"
model_name="qwen"

request_rate=80
num_prompts=1000
result_root="."
burstiness=0.5

for seed in 0 1 2 3 4 5 ; do

  out_dir="${result_root}"
  mkdir -p "$out_dir"

  vllm bench serve \
    --model Qwen/Qwen3.5-9B \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts $num_prompts \
    --request-rate $request_rate \
    --burstiness $burstiness \
    --seed $seed \
    --save-result \
    --save-detailed \
    --metric-percentiles "50,90,95,99" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --result-dir "$out_dir" \
    --result-filename "sharegpt_burstjsp_seed${seed}.json" \
    --port 8000
done
