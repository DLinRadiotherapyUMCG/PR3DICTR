#!/bin/bash

#SBATCH --job-name=HP_resume
#SBATCH --mail-type=END
#SBATCH --mail-user=d.macrae@student.rug.nl
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --output=HP_resume.log


# Install:
#module purge
# NOT NEEDED:
#module load fosscuda/2020b
#module load foss/2022b
#module load OpenCV/4.6.0-foss-2022a-contrib
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 


#module load Python/3.11.3-GCCcore-12.3.0
#python3 -m venv /scratch/$USER/.envs/HNC_env
#source /scratch/$USER/.envs/HNC_env/bin/activate

# NOT NEEDED: 
#module load fosscuda/2020b
#module load foss/2022b
#module load OpenCV/4.6.0-foss-2022a-contrib
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 


## Activate local python environment
module purge
module load Python/3.11.3-GCCcore-12.3.0
source /scratch/$USER/.envs/HNC_env/bin/activate
#pip install numpy --upgrade

# increase limit of open files
ulimit -n 50000

# Train
python3 -u main.py

