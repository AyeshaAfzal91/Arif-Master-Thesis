#!/bin/bash
#SBATCH --job-name=roofline_qwenthree_decode
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --output=logs/roofline_qwenthree_decode_%j.log
#SBATCH --error=logs/roofline_qwenthree_decode_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mehmet.a.bagc@fau.de

source /hnvme/workspace/ihpc170h-pwr_repro/env.sh
module load cuda/13.0.2
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1

cd ~/Arif-Master-Thesis/Start-Task
mkdir -p csv_out

ncu --profile-from-start off --clock-control base  --replay-mode application \
    --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second,dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes.sum.per_second,dram__bytes.sum.peak_sustained,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum.peak_sustained,lts__t_bytes.sum,l1tex__t_bytes.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \
    --csv --page raw --log-file csv_out/qwenthree_decode_s1024_bs1_raw.csv \
    python run_model.py Qwen/Qwen3-4B 1 1024
