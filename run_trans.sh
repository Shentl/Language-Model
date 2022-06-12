#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/logs_trans/%j.out
#SBATCH --error=slurm_logs/logs_trans/%j.err

module load miniconda3
source activate
conda activate common

# training Transforms
# model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
# python -u main.py --cuda --epochs 6 --model Transformer --lr 5 --save ./models/trans/default_trans.pt # 11.60M
# python -u main.py --cuda --epochs 30 --model Transformer --lr 5 --emsize 650 --nhid 650 --save ./models/trans/emsize_nhid_650.pt
python -u main.py --cuda --epochs 30 --model Transformer --lr 5 --emsize 632 --nhid 632 --nhead 8 --nlayers 10 --save ./models/trans/emsize_nhid_632_head8_layer10.pt
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --nhead 10 --nlayers 12 --save ./models/trans/head6_layer12.pt # 14.20M

