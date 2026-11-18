#!/bin/bash
#SBATCH --time=23:58:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --mail-type=END
#SBATCH --mail-user=mehmet.a.bagc@fau.de
#SBATCH --output=logs/litgpt_%x_%j.log
#SBATCH --error=errors/litgpt_%x_%j.err

module load python/3.12-conda
source $HOME/venv_litgpt/bin/activate
module load cuda/13.0.2

cd ~/Arif-Master-Thesis/Start-Task
mkdir -p csv_out/litgpt

SHORT=$1      # e.g. qwen3_litgpt
MODEL=$2      # checkpoint dir, e.g. /hnvme/workspace/ihpc170h-pwr_repro/litgpt_checkpoints/Qwen/Qwen3-4B
PHASE=$3      # prefill or decode

if [ "$PHASE" = "prefill" ]; then
    PROMPT_LEN=4096; OUT_LEN=1; TAG="s4096_bs1"
    WINDOW=""
else
    PROMPT_LEN=1; OUT_LEN=1024; TAG="s1024_bs1"
    WINDOW="-s 650 -c 1000"
fi

ncu --profile-from-start off --clock-control base --replay-mode application $WINDOW \
    --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,dram__cycles_elapsed.avg.per_second,dram__bytes_read.sum,dram__bytes_write.sum,dram__bytes.sum.per_second,dram__bytes.sum.peak_sustained,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum.peak_sustained,lts__t_bytes.sum,l1tex__t_bytes.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \
    --csv --page raw --log-file csv_out/litgpt/${SHORT}_${PHASE}_${TAG}_raw.csv \
    python run_models_litgpt.py "$MODEL" $PROMPT_LEN $OUT_LEN
