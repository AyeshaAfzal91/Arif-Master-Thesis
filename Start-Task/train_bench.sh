#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --mail-type=END
#SBATCH --mail-user=mehmet.a.bagc@fau.de
#SBATCH --output=logs/train_bench_%j.log
#SBATCH --error=errors/train_bench_%j.err

module purge
module load python/3.12-conda
source $HOME/venv_litgpt/bin/activate
module load cuda/13.0.2

cd ~/Arif-Master-Thesis/Start-Task

python train_bench.py "$@"
