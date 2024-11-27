#!/bin/bash

#SBATCH --job-name=HNC_DCNN_ECE
#SBATCH --mail-type=END
#SBATCH --mail-user=d.macrae@student.rug.nl
#SBATCH --time=3-00:00
#SBATCH --partition=gpu
#SBATCH --mem=100G
#SBATCH --gpus-per-node=a100:2
#SBATCH --output=HP_DCNN_ECE.log
#SBATCH --signal=B:SIGUSR1@3600



## Activate local python environment
module purge
module load Python/3.11.3-GCCcore-12.3.0
source /scratch/$USER/.envs/HNC_env/bin/activate
#pip install numpy --upgrade

# increase limit of open files
ulimit -n 500000

# Train
python3 -u main_ens.py

