#!/bin/bash
declare -A MODELS=(
  [llama]="nvidia/Llama-3.1-Minitron-4B-Depth-Base"
  [nemotron]="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
  [qwenthree]="Qwen/Qwen3-4B"
  [qwenthree_half]="Qwen/Qwen3.5-4B"
)

mkdir -p logs errors

for short in "${!MODELS[@]}"; do
  model="${MODELS[$short]}"
  for phase in prefill decode; do
    sbatch --job-name="roofline_${short}_${phase}" \
           --output="logs/roofline_${short}_${phase}_%j.log" \
           --error="errors/roofline_${short}_${phase}_%j.err" \
           profile_run.sh "$short" "$model" "$phase"
  done
done
