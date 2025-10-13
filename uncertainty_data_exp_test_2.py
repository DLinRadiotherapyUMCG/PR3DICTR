import copy
from src.experiments.experimentHandler import experimentHandler
from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

import src.hyper_opt.WandB_functions as WandB_functions
import os
import torch, gc

if __name__ == '__main__':

    # Setup
    log_level = parse_args()
    setup_logging(log_level)
    
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    CONFIG_NAME = 'Daniel/Uncertainty_main'
    config = get_config(CONFIG_NAME)

    # Disable randomness
    set_random_seed(config['general']['seed'])

    using_WandB = config['hyperparam_tuning']['WandB']['isEnabled']
        
    if (using_WandB):
        print("logging in..........")
        # Log into weights and biases
        WandB_functions.login(config)
        print("Weights and Biases is active")
    

    from src.uncertainty.deep_ensemble import train_deep_ensemble_models, evaluate_deep_ensemble_models
    from src.uncertainty.MC_dropout import train_MC_dropout_model, collect_bayesian_forward_passes, post_hoc_collect_bayesian_forward_passes
    
    from uncertainty_main import load_model_config_for_uncertainty_experiment
    
    # ["Geslacht", "Leeftijd", "Dysphagia_W01_Grade0_1", "Dysphagia_W01_Grade2", "Dysphagia_W01_Grade3_4"]
    
    # config['general']['dataset_amounts_experiment'] = False

    

    training_patients_dict = {
        "OS" : [50, 100, 150, 200], 
        "LRC" : [50, 100, 150],
        "Dysphagia_M06" : [100, 200, 300, 400, 500, 600, 700],
        "Xerostomia_M06" : [100, 200, 300, 400, 500, 600, 700],
        "Metal_artefact" : [10, 50, 100, 200, 500]
    }

    endpoints = ["Dysphagia_M06", "Xerostomia_M06", "OS", "LRC"]

    endpoints = ["OS", "Dysphagia_M06",  "Xerostomia_M06", "LRC"]

    endpoints = ["OS"]

    endpoints = ['Metal_artefact']
    endpoints = ["Dysphagia_M06", "Xerostomia_M06"]
    endpoints = ['Xerostomia_M06']

    for endpoint in endpoints:
        for idx in range(5):
        
            run_config = load_model_config_for_uncertainty_experiment(copy.deepcopy(config), endpoint_name=endpoint)
        
            #config['general']['experiment_name'] = "Data MC Dropout"
            run_config['general']['trialNumber'] = endpoint + f"_run_{idx+1}"


            run_config['general']['dataset_amounts_experiment'] = True
            run_config['data']['n_training_patients_list'] = training_patients_dict[endpoint] # , 100, 150] #  [100, 200, 300, 400, 500, 600, 700, 800] # 
            
            set_random_seed(idx)
            # run_config['general']['experiment_name'] = "Data MC Dropout PRIMA no Aug"
            # run_config['data']['augmentation']['isEnabled'] = False
            run_config['general']['experiment_name'] = "Data MC Dropout PRIMA mini_model"
            train_MC_dropout_model(run_config, UQ_method="MC_dropout")

            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()


    for endpoint in endpoints:
        for idx in range(5):
        
            run_config = load_model_config_for_uncertainty_experiment(copy.deepcopy(config), endpoint_name=endpoint)
        
            #config['general']['experiment_name'] = "Data MC Dropout"
            run_config['general']['trialNumber'] = endpoint + f"_run_{idx+1}"
            
            run_config['general']['dataset_amounts_experiment'] = True
            run_config['data']['n_training_patients_list'] = training_patients_dict[endpoint] # , 100, 150] #  [100, 200, 300, 400, 500, 600, 700, 800] # 
            
            # if not endpoint == "Dysphagia_M06":
            set_random_seed(idx) 
            run_config['general']['experiment_name'] = "Data TTA PRIMA mini_model"
            train_MC_dropout_model(run_config, UQ_method="TTA")

            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()


    for endpoint in endpoints:
        for idx in range(5):
        
            run_config = load_model_config_for_uncertainty_experiment(copy.deepcopy(config), endpoint_name=endpoint)
        
            #config['general']['experiment_name'] = "Data MC Dropout"
            run_config['general']['trialNumber'] = endpoint + f"_run_{idx+1}"
  
            
            run_config['general']['dataset_amounts_experiment'] = True
            run_config['data']['n_training_patients_list'] = training_patients_dict[endpoint] # , 100, 150] #  [100, 200, 300, 400, 500, 600, 700, 800] # 
            
            # if not endpoint == "Dysphagia_M06":
            set_random_seed(idx)
            run_config['general']['experiment_name'] = "Data Deep Ensemble PRIMA mini_model"
            run_config['uncertainty']['deep_ensemble']['n_models'] = 5         # NOTE: only 5 models, 10 takes too long for the data experiment
            train_deep_ensemble_models(run_config)
            #evaluate_deep_ensemble_models(run_config)

            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()



