#!/bin/bash

#SBATCH --job-name=HNC_make_venv
#SBATCH --mail-type=END
#SBATCH --mail-user=d.macrae@student.rug.nl  # email updates don't work on habrok yet
#SBATCH --time=1:59:59              # shorter time, this script just installs libraries
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --output=make_venv.log


# Install:
module purge
#module load fosscuda/2020b
#module load foss/2022b
module load Python/3.11.3-GCCcore-12.3.0              # import Python
#module load OpenCV/4.6.0-foss-2022a-contrib
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 
python3 -m venv /scratch/$USER/.envs/HNC_env             # make a .venv with all the dependencies needed
source /scratch/$USER/.envs/HNC_env/bin/activate
pip install torch torchvision torchaudio torcheval
pip install torchinfo tqdm monai pytz SimpleITK pydicom scikit-image matplotlib numpy 
pip install torch_optimizer torcheval "ray[tune]" optuna ray
pip install scikit-learn opencv-python tabulate
pip install timm pandas plotly nbformat colorcet einops toml wandb
pip install --upgrade pip
pip install numpy --upgrade


