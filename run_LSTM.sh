#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/logs_LSTM/%j.out
#SBATCH --error=slurm_logs/logs_LSTM/%j.err

module load miniconda3
source activate
conda activate common

# training LSTM
# python -u main.py --cuda --epochs 30 --lr 5 --emsize 640 --nhid 600 --nlayers 8 --save ./models/LSTM/emsize_nhid_640_600_layer8.pt
# python -u main.py --cuda --epochs 40 --lr 5 --emsize 650 --nhid 650 --nlayers 4 --dropout 0.5 --save ./models/LSTM/emsize_nhid_650_nlayer_4_dropout_05.pt
# python -u main.py --cuda --epochs 50 --lr 5 --emsize 650 --nhid 650 --nlayers 3 --dropout 0.5 --save ./models/LSTM/emsize_nhid_650_nlayer_3_dropout_05.pt
# python -u main.py --cuda --epochs 40 --lr 5 --emsize 720 --nhid 720 --nlayers 2 --dropout 0.5 --save ./models/LSTM/emsize_nhid_720_dropout_05.pt
# python -u main.py --cuda --epochs 40 --lr 5 --emsize 900 --nhid 840 --dropout 0.5 --save ./models/LSTM/emsize_nhid_900_840_dropout_06.pt
# python -u main.py --cuda --epochs 50 --lr 5 --emsize 900 --nhid 840 --dropout 0.65 --save ./models/LSTM/emsize_nhid_900_840_dropout_065.pt
# python -u main.py --cuda --epochs 40 --lr 5 --emsize 700 --nhid 950 --nlayers 2 --dropout 0.5 --save ./models/LSTM/emsize_nhid_700_950_dropout_05.pt
# Try tensorboard
# python -u main.py --cuda --epochs 6 --lr 5 --emsize 100 --nhid 100 --use_tensorboard --save ./models/LSTM/try.pt
# python -u main.py --cuda --epochs 40 --lr 5 --emsize 600 --nhid 1100 --nlayers 2 --dropout 0.2 --save ./models/LSTM/emsize_nhid_600_1100_dropout_02.pt
# python -u main.py --cuda --epochs 40 --lr 5 --emsize 600 --nhid 1100 --nlayers 2 --dropout 0.3 --save ./models/LSTM/emsize_nhid_600_1100_dropout_03.pt
# python -u main.py --cuda --epochs 40 --lr 5 --emsize 600 --nhid 1100 --nlayers 2 --dropout 0.4 --save ./models/LSTM/emsize_nhid_600_1100_dropout_04.pt
# python -u main.py --cuda --epochs 50 --lr 6.5 --emsize 900 --nhid 840 --dropout 0.5 --save ./models/LSTM/em_nhid_900_840_drop_05_lr65.pt
# python -u main.py --cuda --epochs 50 --lr 7 --emsize 900 --nhid 840 --dropout 0.5 --save ./models/LSTM/em_nhid_900_840_drop_05_lr7.pt
# python -u main.py --cuda --epochs 50 --lr 7.5 --emsize 900 --nhid 840 --dropout 0.5 --save ./models/LSTM/em_nhid_900_840_drop_05_lr75.pt
# python -u main.py --cuda --epochs 55 --lr 8 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr8.pt
# python -u main.py --cuda --epochs 55 --lr 9 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr9.pt
# python -u main.py --cuda --epochs 55 --lr 10 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr10.pt
# python -u main.py --cuda --epochs 55 --lr 8 --emsize 900 --nhid 840 --dropout 0.3 --save ./models/LSTM/em_nhid_900_840_drop_03_lr8.pt
# python -u main.py --cuda --epochs 50 --lr 11 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr11.pt
# python -u main.py --cuda --epochs 50 --lr 12 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr12.pt
# python -u main.py --cuda --epochs 50 --lr 15 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr15.pt
# python -u main.py --cuda --epochs 50 --lr 11 --emsize 900 --nhid 840 --dropout 0.35 --save ./models/LSTM/em_nhid_900_840_drop_035_lr11.pt
# python -u main.py --cuda --epochs 50 --lr 12 --emsize 900 --nhid 840 --dropout 0.35 --save ./models/LSTM/em_nhid_900_840_drop_035_lr12.pt
# python -u main.py --cuda --epochs 50 --lr 15 --emsize 900 --nhid 840 --dropout 0.35 --save ./models/LSTM/em_nhid_900_840_drop_035_lr15.pt
# python -u main.py --cuda --epochs 50 --lr 15 --emsize 900 --nhid 840 --dropout 0.5 --save ./models/LSTM/em_nhid_900_840_drop_05_lr15.pt
# python -u main.py --cuda --epochs 50 --lr 20 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr20.pt
# python -u main.py --cuda --epochs 50 --lr 13 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr13.pt
# python -u main.py --cuda --epochs 50 --lr 14 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr14.pt
# python -u main.py --cuda --epochs 50 --lr 16 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr16.pt
# python -u main.py --cuda --epochs 50 --lr 18 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr18.pt
# python -u main.py --cuda --epochs 50 --batch_size 32 --lr 13 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr13.pt
# python -u main.py --cuda --epochs 50 --batch_size 64 --lr 13 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr13.pt
# python -u main.py --cuda --epochs 50 --batch_size 128 --lr 13 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr13.pt
# python -u main.py --cuda --epochs 50 --batch_size 256 --lr 13 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr13.pt
# python -u main.py --cuda --epochs 50 --batch_size 32 --lr 10 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr10_b32.pt
# python -u main.py --cuda --epochs 50 --batch_size 64 --lr 4 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr4_b64.pt
# python -u main.py --cuda --epochs 50 --batch_size 64 --lr 7 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr7_b64.pt
# python -u main.py --cuda --epochs 50 --batch_size 64 --lr 10 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr10_b64.pt
python -u main.py --cuda --epochs 50 --batch_size 64 --lr 15 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr15_b64.pt
python -u main.py --cuda --epochs 50 --batch_size 64 --lr 17 --emsize 900 --nhid 840 --dropout 0.4 --save ./models/LSTM/em_nhid_900_840_drop_04_lr17_b64.pt