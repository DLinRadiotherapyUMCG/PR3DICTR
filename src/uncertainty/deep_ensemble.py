import os
import logging
import pandas as pd
import numpy as np
import torch
import gc

from src.constants import DEVICE

from src.dataset.load_dataset import load_dataset, generate_K_fold_cross_validation_splits
from src.models.tools.get_classification_model import get_classification_model
from src.dataset.get_dataloader import make_dataloader   
from src.uncertainty.utils import set_dataset_config_for_uncertainty
from src.utils.set_random_seed import set_random_seed
from src.dataset.get_transforms import get_transforms
from src.utils.loss_func.get_loss_function import get_loss_function
from src.evaluation.mainMetricHandler import mainMetricHandler
from src.utils.saving.create_results_directory import create_results_directory
from src.config_presets.tools.save_config import save_config
from src.models.tools.get_classification_model import get_classification_model
from src.training.train import train
from src.training.validate import validate



def get_dataset_for_uncertainty_experiments(config):
    # Load the dataset and dataloader
    # load the data, and make K-fold splits
    df_train_val, df_test = load_dataset(config)
    k_fold_dataframes_list = generate_K_fold_cross_validation_splits(config, df_train_val)
    train_transforms, val_transforms = get_transforms(config)

    dataset_split_dict = k_fold_dataframes_list[0]  # the first split is used for training and validation
    train_data, val_data = dataset_split_dict['train'], dataset_split_dict['val']

    train_loader, metadata = make_dataloader(config, train_data, train_transforms, validation_mode=False)
    val_loader, _ = make_dataloader(config, val_data, val_transforms, validation_mode=True)
    test_loader, _ = make_dataloader(config, df_test, val_transforms, validation_mode=True)

    return train_loader, val_loader, test_loader, metadata



def train_deep_ensemble_model(config):
    """
    Train a deep ensemble model.
    """
    
    set_random_seed(config['general']['seed'])  # Set the random seed for reproducibility
    


    """
    Make the dataset and dataloader for uncertainty experiments.
    """
    # override the dataloader settings
    set_dataset_config_for_uncertainty(config)

    # Load the dataset and dataloader
    train_loader, val_loader, test_loader, metadata = get_dataset_for_uncertainty_experiments(config)
    
    # get the loss function and metric handler
    loss_function = get_loss_function(config)
    metricHandler = mainMetricHandler(config)  # class to compute the metrics per epoch

    """
    Train each model in the ensemble. 
    Each model will be trained with a different random seed, but the same training & validation data.
    The results are be saved in separate folders for each model.
    """

    n_models = config['uncertainty']['deep_ensemble']['n_models']
    # generate a random seed for each model
    seeds = [np.random.randint(0, 10000) for _ in range(n_models)]

    for model_idx in range(1, n_models + 1):
        set_random_seed(seeds[model_idx - 1])  # Set the random seed for each model
        print(model_idx)

        # set the directory for this model
        create_results_directory(config, folder_name=f"model_{model_idx}")

        # save the config file for this model
        save_config(config)

        print(config['general']['resultsCurrentDirectory'])


        logging.info('Getting model')
        model = get_classification_model(config, metadata=metadata, save_summary=True)
        model.to(device=DEVICE)
        model = train(config, model, loss_function, train_loader, val_loader, metricHandler)

    pass  # Replace with actual implementation


