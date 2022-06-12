#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/logs_RNN/%j.out
#SBATCH --error=slurm_logs/logs_RNN/%j.err

module load miniconda3
source activate
conda activate common

# training RNN
python -u main.py --cuda --epochs 45 --model RNN_RELU --lr 5 --emsize 900 --nhid 840 --dropout 0.5 --save ./models/RNN/RELU_em900_hid840_drop05.pt

