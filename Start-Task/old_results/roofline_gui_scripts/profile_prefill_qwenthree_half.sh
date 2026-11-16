#!/bin/bash
#SBATCH --job-name=profile_prefill_qwenthree_half
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/profile_qwenthree_half_prefill_%j.log
#SBATCH --error=errors/profile_qwenthree_half_prefill_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mehmet.a.bagc@fau.de

source /hnvme/workspace/ihpc170h-pwr_repro/env.sh
module load cuda/13.0.2
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1  

cd ~/Arif-Master-Thesis/Start-Task

ncu --profile-from-start off --clock-control base \
    --section SpeedOfLight_RooflineChart \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --export prefill_qwenthree_half.ncu-rep \
    python run_model.py Qwen/Qwen3.5-4B 4096 1
