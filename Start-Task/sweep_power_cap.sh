#!/bin/bash
#SBATCH --time=00:58:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --mail-type=END
#SBATCH --mail-user=mehmet.a.bagc@fau.de

source /hnvme/workspace/ihpc170h-pwr_repro/env.sh
module load cuda/13.0.2
export HF_HUB_OFFLINE=1
cd ~/Arif-Master-Thesis/Start-Task

MODEL=$1        # e.g.  nvidia/Llama-3.1-Minitron-4B-Depth-Base
PROMPT_LEN=$2   # e.g.  1
OUT_LEN=$3      # e.g.  1024

for cap in 280 420 500 600 700; do
    echo "=== power cap: ${cap}W ==="
    nvidia-smi -pl $cap
    python energy_profile.py "$MODEL" "$PROMPT_LEN" "$OUT_LEN" "cap_${cap}"
done

nvidia-smi -pl 700   # sweep bitince varsayilana geri don
