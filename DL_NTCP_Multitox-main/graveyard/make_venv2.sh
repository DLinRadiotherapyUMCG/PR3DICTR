#!/bin/bash

#SBATCH --job-name=Xerostomia_1
#SBATCH --mail-type=END
#SBATCH --mail-user=d.macrae@student.rug.nl  # email updates don't work on habrok yet
#SBATCH --time=1:59:59              # shorter time, this script just installs libraries
#SBATCH --output=slurm-venv2-setup.log


# Install:
module purge
#module load fosscuda/2020b
#module load foss/2022b
module load Python/3.11.3-GCCcore-12.3.0              # import Python
#module load OpenCV/4.6.0-foss-2022a-contrib
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 
python3 -m venv $HOME/venvs/HNC_notebook

source /scratch/$USER/.envs/HNC_notebook/bin/activate
pip install --upgrade pip wheel setuptools
pip install tomlkit jupyter-contrib-core

pip install torch torchvision torchaudio
pip install torchinfo tqdm monai pytz SimpleITK pydicom scikit-image matplotlib numpy 
pip install torch_optimizer torcheval "ray[tune]" optuna
pip install scikit-learn opencv-python
pip install timm pandas
pip install --upgrade pip
pip install numpy --upgrade

python3 -m ipykernel install --user --name=HNC_notebook