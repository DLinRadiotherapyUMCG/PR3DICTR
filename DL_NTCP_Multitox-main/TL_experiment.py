import os
import sys
from datetime import datetime
import argparse

from train_autoencoder import main as train_autoencoder_main

from data_preproc.data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from misc import make_config_object
from main import perform_N_fold_cross_validation


def initialize(config, device, create_folders=True):
    """
    Set up experiment and save the experiment path to be able to save results.
    Args:
        torch_version: the version of pytorch (just for printing it in the logger)
        device: the device to run the model on (also for printing)
        create_folders: whether or not to create the folders for saving the results
    Returns:
        logger: the logger object

    """
    if create_folders:
        # Create experiment folder and subfolders if they do not exist yet
        exp_dir = config.exp_dir
        #exp_figures_dir = os.path.join(exp_dir, 'figures')
        create_folder_if_not_exists(exp_dir)
        #create_folder_if_not_exists(exp_figures_dir)

        # Logger output filename
        output_filename = os.path.join(exp_dir, 'log.txt')
           
    else:
        output_filename = None

    # Initialize logger
    logger = Logger(output_filename=output_filename, suppress_CLI_prints=True)

    # Print variables
    logger.my_print(f'Device: {device}.')
    logger.my_print(f'Torch version: {config.torch_version}.')
    logger.my_print(f'Torch.backends.cudnn.benchmark: {config.cudnn_benchmark}.')

    return logger





def main(train_model_name):
    

    #folder_name = "Masked_Autoencoder_0"

    #models = ["dcnn_pooling","resnet_lrelu", "ViT", "resnext_lrelu"]
    #models = ["ViT", "resnext_lrelu", "convnext"]   
    #models = [train_model_name]

    
    import config as config_file

    config_file.perform_test_run = False 
    if config_file.perform_test_run:
        config_file.n_samples = 400
        config_file.max_epochs = 1
        config_file.perform_data_aug = False
        config_file.perform_augmix = False
        config_file.cv_folds = 2

    # override config file for the autoencoder training settings
    config_file.model_name = train_model_name
    config_file.use_best_config = True
    # set export dirs
    autoencoder_exp_dir = os.path.join(config_file.root_path, 'TL_experiments', train_model_name, "autoencoder_model")
    autoencoder_model_weights_path = os.path.join(autoencoder_exp_dir, config_file.filename_best_model_pth)
    config_file.exp_dir = autoencoder_exp_dir

    
    model_config = make_config_object(config_file)  # deals with all of the 'best' model parameters

    # make logger
    logger = initialize(model_config, device=model_config.device, create_folders=True)
    """
    TRAIN AUTOENCODER MODEL
    """

    #train_autoencoder_main(train_model_name, logger, override_config=model_config, TEST_MODE = config_file.perform_test_run)

    print("TRAINED AUTOENCODER MODEL: ", train_model_name)
    print("WEIGHTS:", autoencoder_model_weights_path)


    """
    TL FOR 5 FOLDS (MAIN CODE)
    """


    model_config.pretrained_path = autoencoder_exp_dir
    model_config.transfer_learning_mode = True
    model_config.lr = model_config.lr/10
    

    TL_exp_dir = os.path.join(config_file.root_path, 'TL_experiments', train_model_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    autoencoder_model_weights_path = os.path.join(autoencoder_exp_dir, config_file.filename_best_model_pth)
    model_config.exp_dir = TL_exp_dir
    model_config.exp_figures_dir = os.path.join(TL_exp_dir, 'figures')
    model_config.temp_model_weights_dir = os.path.join(TL_exp_dir, 'tmp_model_weights')
    create_folder_if_not_exists(TL_exp_dir)

    print("PERFORMING TRANSFER LEARNING FOR 5 FOLDS")
    perform_N_fold_cross_validation(cfg=model_config, main_logger=logger)








if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', type=str, help='Name of the train model', nargs="*")
    args = parser.parse_args()

    train_model_names = args.model_names

    for train_model_name in train_model_names:

        main(train_model_name)
    