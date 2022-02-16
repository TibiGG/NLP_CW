#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tg4018

export PATH=/vol/bitbucket/tg4018/anaconda3/envs/nlp/bin/:$PATH
conda activate nlp
. /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100
/usr/bin/nvidia-smi

cd /vol/bitbucket/tg4018/NLP_CW
python trainer.py
uptime
