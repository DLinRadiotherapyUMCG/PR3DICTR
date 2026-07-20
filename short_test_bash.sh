#!/bin/bash

#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

module load Python/3.9.6-GCCcore-11.2.0
source /scratch/hb-LungModeling/Python_Environments/DL_ModellingV2/bin/activate
#pip install einops
#pip install lifelines
#pip install ruamel.yaml

#cd /scratch/hb-LungModeling/CodeDLPOS/DLModelling_Code_2025_12

## Python script to run
python -u ./test_configs.py -p "HP_TransRP" 
