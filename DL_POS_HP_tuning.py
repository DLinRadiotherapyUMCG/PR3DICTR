import os
import sys
import pathlib
import argparse
path_src = os.getcwd()
sys.path.insert(1, path_src)

from src.config_presets.tools.get_config import get_config
from src.config_presets.tools.load_config import load_config
from src.config_presets.tools.check_update_config import update_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.experiments.experimentHandler import experimentHandler
from src.evaluation.validate_on_test_set import validate_models_on_test_set



def merge_DL_POS_configs(model_specific_config_path, main_config_name="main_Lung_PET_modelling_config"):

    config = get_config(main_config_name)  # load the main config
    total_search_space = config['hyperparam_tuning']['hyperparams'].copy() # store a copy of the hyperparameter search space
    print("main search space: ", total_search_space)


    # get the model-specific config
    experiment_config = load_config(model_specific_config_path) # load the model-specific config from the provided path
    model_specific_hyperparam_search_space = experiment_config['hyperparam_tuning']['hyperparams'].copy() # also a copy of this hyperparameter search space
    print("model-specific search space: ", model_specific_hyperparam_search_space)
    
    # merge the configs
    update_config(base=config, updates=experiment_config)    # merge the configs
    total_search_space.update(model_specific_hyperparam_search_space) # merge the hyperparameter search spaces

    config['hyperparam_tuning']['hyperparams'] = total_search_space # update the merged hyperparameter search space 

    print("Merged main and model-specific hyperparameter search space configs !")
    
    return config



if __name__ == '__main__':

    # Setup
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str,required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    path_to_config = str(pathlib.Path(args.path))
    print("Config file is ",path_to_config)
    log_level = args.log_level
    setup_logging(log_level)

    # Merge the main DL POS config with the model-specific experiment configs!
    config = merge_DL_POS_configs(model_specific_config_path=path_to_config)

    # run experiment
    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)