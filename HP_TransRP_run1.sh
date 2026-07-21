#!/bin/bash

#SBATCH --job-name=HP_TransRP_Run1
#SBATCH --output=HP_TransRP_Run1.log
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

module load Python/3.9.6-GCCcore-11.2.0
source /scratch/hb-LungModeling/Python_Environments/DL_ModellingV2/bin/activate

#cd /scratch/hb-LungModeling/CodeDLPOS/DLModelling_Code_2025_12


## Python script to run
python -u ./DL_POS_HP_tuning.py -p "HP_DenseNet_Run1" 
