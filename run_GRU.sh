#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/logs_GRU/%j.out
#SBATCH --error=slurm_logs/logs_GRU/%j.err

module load miniconda3
source activate
conda activate common

# training GRU
# model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
python -u main.py --cuda --epochs 30 --model GRU --lr 5 --emsize 632 --nhid 632 --nhead 8 --nlayers 10 --save ./models/trans/emsize_nhid_632_head8_layer10.pt
