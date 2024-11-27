import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset, load_dataset_total
from src.models.tools.save_model import save_model
from src.training.train_multi import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file, create_textfile


from src.utils.loss_func.get_loss_function import get_loss_function
from src.dataset.get_dataloader import make_dataloader   
from src.dataset.get_transforms import get_transforms


if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Multi_tox')

    # Disable randomness
    set_random_seed(config['general']['seed'])

    # # MAIN: DL running class with hyperparameter optimization
    hyperClass = HyperTuning_Handler(config)
    hyperClass.Operate(config)
    hyperClass.Stop()

    # # Set loss function
    # loss_function = get_loss_function(config)

    # # Create the dataset dataframes
    # DFs_col = load_dataset_total(config)
    # trainDF_col = DFs_col[0]
    # valDF_col= DFs_col[1]
    # testDF_col = DFs_col[2]

    # # make the training and validation transforms
    # train_transforms, val_transforms = get_transforms(config)


    # train_loader, metadata = make_dataloader(config, trainDF_col[0], train_transforms, validation_mode=False)
    # val_loader, _ = make_dataloader(config, valDF_col[0], val_transforms, validation_mode=True)

   
    # model = train(config, train_loader, val_loader, metadata)
    