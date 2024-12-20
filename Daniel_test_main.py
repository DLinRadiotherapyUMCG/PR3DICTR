import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.experiments.experimentHandler import experimentHandler
import src.hyper_opt.WandB_hpt as WandB_hpt


from src.utils.fileHandler import create_file, create_textfile


from src.utils.loss_func.get_loss_function import get_loss_function
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.get_transforms import get_transforms

from src.training.k_fold_cross_validation import K_fold_cross_validation

#from src.hyper_opt.WandB_hpt import login

import numpy as np
import random
import torch
from monai.utils import set_determinism

if __name__ == '__main__':
    # Setup
    log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Multi_tox')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    seed = config['general']['seed']
    # Set seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    from src.dataset.get_transforms import get_random_augmentation_names_from_config

    #transforms = get_random_augmentations_from_config(config)


    


    #K_fold_cross_validation(config)

    # # MAIN: DL running class with hyperparameter optimization
    #
    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)
    #####hyperClass.Stop()


    # TEST ENSEMBLE CODE

    # from src.evaluation.validate_on_test_set import validate_models_on_test_set


    # trial_dir = r"C:/Users/S.P.M. de Vette/OneDrive - UMCG/Desktop/pred_RT_results/test_eval/Trial_3" # config['general']['resultsCurrentDirectory']
    # # run the models on the test set
    # validate_models_on_test_set(config, trial_dir)

