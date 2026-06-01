#!/bin/bash
set -euo pipefail

model="Qwen/Qwen3.5-9B"
model_name="qwen"

request_rate=80
num_prompts=1000
input_len=128
output_len=128
result_root="."

for burstiness in 0.5; do

  out_dir="${result_root}"
  mkdir -p "$out_dir"

  vllm bench serve \
    --model Qwen/Qwen3.5-9B \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len $input_len \
    --random-output-len $output_len \
    --ignore-eos \
    --num-prompts $num_prompts \
    --request-rate $request_rate \
    --burstiness $burstiness \
    --seed 0 \
    --save-result \
    --save-detailed \
    --metric-percentiles "50,90,95,99" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --result-dir "$out_dir" \
    --result-filename "test_rate80burst0.5burstrr.json" \
    --port 8000
    #--plot-timeline \
done
