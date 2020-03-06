#!/bin/bash
#SBATCH --time=00-03:00                       # time (DD-HH:MM) 
#SBATCH --gres=gpu:1                         # Number of GPUs (per node)
#SBATCH --mem=16G                            # memory per node -> use powers of 2 
#SBATCH --cpus-per-task=1                    # CPU cores/threads
#SBATCH --qos=normal                        
#SBATCH --partition=gpu

eval "$(conda shell.bash hook)"
conda activate 401a2

TRAIN=/scratch/ssd001/home/cchoquet/csc401/a2/data/Hansard/Training/
TEST=/scratch/ssd001/home/cchoquet/csc401/a2/data/Hansard/Testing/
python a2_run.py vocab $TRAIN e vocab.e.gz
python a2_run.py vocab $TRAIN f vocab.f.gz
python a2_run.py split $TRAIN train.txt.gz dev.txt.gz
python a2_run.py train $TRAIN 5 vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_wo_att.pt.gz --device cuda
python a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_w_att.pt.gz --with-attention --device cuda
python a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_w_att.pt.gz --with-attention --device cuda
python a2_run.py test $TEST vocab.e.gz vocab.f.gz model_w_att.pt.gz --with-attention --device cuda
