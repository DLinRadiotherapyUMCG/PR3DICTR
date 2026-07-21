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




if __name__ == '__main__':

    # Setup
    # parser = argparse.ArgumentParser(description='Model Training Script')
    # parser.add_argument("-p", "--path", type=str,required=True)
    # parser.add_argument("--log-level", type=str, default="INFO")
    # args = parser.parse_args()
    # path_to_config = str(pathlib.Path(args.path))
    # print("Config FIle is ",path_to_config)
    # log_level = args.log_level
    # setup_logging(log_level)



    # config = get_config('main_Lung_PET_modelling_config')  
    # total_search_space = config['hyperparam_tuning']['hyperparams'].copy()

    # # get the model-specific config
    # #experiment_config = load_config('HP_TransRP')  # Load the config for the HNC OS model
    # experiment_config = load_config(path_to_config) 
    # model_specific_hyperparam_search_space = experiment_config['hyperparam_tuning']['hyperparams'].copy()
    
    
    # print('config 1')
    # print(total_search_space)

    # print('\n\n\n experiment_config')
    # print(model_specific_hyperparam_search_space)

    # print('\n\n\n')

    # # merge the configs
    # update_config(base=config, updates=experiment_config)

    # print('config 2')
    # # merge the hyperparameter search spaces
    # print(config['hyperparam_tuning']['hyperparams'])
    
    # total_search_space.update(model_specific_hyperparam_search_space)

    # config['hyperparam_tuning']['hyperparams'] = total_search_space

    
    # print('done')
    # print(config['hyperparam_tuning']['hyperparams'])

    # print('\n\n\n')
    # print(total_search_space)

    # print(type(total_search_space))



    # expHandler = experimentHandler(config)
    # expHandler.run_experiment(config)


    import numpy as np

    path = "/scratch/hb-LungModeling/Data/DLPOS/PET_Projects_Imaging_v2_SameSize/3316002/segmentation_map.npy"
    array = np.load(path)
    print(array.shape)
    print(np.unique(array))

    