# Word-level Language Modeling using RNN and Transformer
### 环境配置
并没有用到额外的包，只需要将该任务原本配好的环境重命名为common就行

或者运行如下脚本
```bash
conda env create -f environment.yml
conda activate common
```
### 运行
#### 最佳性能复现
```angular2html
python -u main.py --cuda --epochs 40 --model LSTM --batch_size 32 --lr 18 --emsize 1250 --nhid 1250 --nlayers 2 --dropout 0.45 --tied --use_tensorboard
```
#### 复现报告中的各种尝试结果
在下面4个脚本中使用报告中的各项实验参数
```bash
sbatch run_GRU.sh
sbatch run_LSTM.sh
sbatch run_RNN.sh
sbatch run_trans.sh
```
#### Tensorboard
Tensorboard结果在./runs中

如果要使用Tensorboard，只需在脚本中加入
```angular2html
--use_tensorboaed
```
