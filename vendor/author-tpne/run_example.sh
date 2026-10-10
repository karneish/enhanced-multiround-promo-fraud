#!/bin/sh

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11/3.11.9 cuda/12.1/12.1.1 gcc/13.2.0
source ~/venv/cuda_torch_dgl/bin/activate
cd scripts
python3 main.py -c $1
deactivate