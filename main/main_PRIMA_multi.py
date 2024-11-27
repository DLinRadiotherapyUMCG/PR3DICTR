import logging
import os

import wandb

# Set working directory to --> "Pred_RT"
import sys
from pathlib import Path
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.dataset.load_dataset import load_dataset_total
from src.models.tools.save_model import save_model
from src.training.train import train, validate
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from src.hyper_opt.hyperHandler import HyperTuning_Handler
from src.utils.fileHandler import create_file
from src.training.train import move_batch_to_device
from src.constants import DEVICE
from src.models.tools.get_classification_model import get_classification_model

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


    

    # MAIN: DL running class with hyperparameter optimization
    # hyperClass = HyperTuning_Handler(config)
    # hyperClass.Operate(config)
    # hyperClass.Stop()

    # Load the dataset
    datasets_col = load_dataset_total(config)
    trainDataset_col = datasets_col[0]
    valDataset_col= datasets_col[1]
    testDataset_col = datasets_col[2]

    train_transforms, val_transforms = get_transforms(config)

    # Train and time a model
    for i in range(len(trainDataset_col)):

        train_loader, metadata = make_dataloader(config, trainDataset_col[i], train_transforms, validation_mode=False)
        print(metadata)
        val_loader, metadata2 = make_dataloader(config, valDataset_col[i], val_transforms, validation_mode=True)
        print(metadata2)
        # OLD
        # train_loader = DataLoader(trainDataset_col[i], batch_size=config['training']['batch_size'], shuffle=True, 
        #                               num_workers = config['data']['dataloader']['num_workers'], persistent_workers = config['data']['dataloader']['persistent_workers'])
        # val_loader = DataLoader(valDataset_col[i], batch_size=config['training']['batch_size'], shuffle=False,
        #                          num_workers = 1, persistent_workers = config['data']['dataloader']['persistent_workers'])
        
        # model = get_classification_model(config, metadata)
        # model.to(device=DEVICE)
        # model.eval()

        # loss_function = get_loss_function(config)

        # validate(loss_function, model, val_loader, config)
        
        model = train(config, train_loader, val_loader, metadata, hyperClass = None)

        


def targets_to_dict(targets, labels_list):
    targets_dict = dict()
    for i, label in enumerate(labels_list):
        targets_dict[label] = targets[:, i]
    return targets_dict