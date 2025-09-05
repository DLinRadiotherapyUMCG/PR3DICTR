from src.config_presets.tools.get_config import get_config, update_config
from src.config_presets.tools.load_config import load_config

from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

import os

from src.experiments.experimentHandler import experimentHandler
import src.hyper_opt.WandB_functions as WandB_functions


def load_modal_config_for_uncertainty_experiment(config, endpoint_name = "Dysphagia_M06"):
    if endpoint_name == "OS":
        model_config = load_config("Daniel/uncertainty_models/Baoqiang_OS")
        
    elif endpoint_name == "LRC":
        model_config = load_config("Daniel/uncertainty_models/Baoqiang_LRC")
    elif endpoint_name == "Dysphagia_M06":
        model_config = load_config("Daniel/uncertainty_models/Suzanne_Dysphagia")
    elif endpoint_name == "Xerostomia_M06":
        model_config = load_config("Daniel/uncertainty_models/Hung_Xerostomia")
    else:
        raise ValueError("Endpoint not recognized. Please choose either 'OS', 'LRC', 'Dysphagia_M06', or 'Xerostomia_M06'.")
    
    # use the model_config to adjust the model/endpoints/features in the main config
    update_config(base=config, updates=model_config, allow_new_keys=False)

    # update labels
    from src.dataset.LabelTypesManager import LabelTypesManager
    labelManager = LabelTypesManager(config=config)  # get the label types manager from the config
    config['saving']['label_column_names'] = labelManager.label_names_full_list

    return config



if __name__ == '__main__':
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()


    log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config


    # First load in the main config for all uncertainty experiments
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
    from src.uncertainty.MC_dropout import train_MC_dropout_model, collect_bayesian_forward_passes
    
    # Choose the endpoint
    endpoint_list = ["Dysphagia_M06", "Xerostomia_M06", "OS", "LRC"]

    config['general']['experiment_name'] = "Normal UQ Models"

    for endpoint in endpoint_list:
        config = get_config(CONFIG_NAME)

        # Disable randomness
        set_random_seed(config['general']['seed'])
    

        config = load_modal_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

        # SETTINGS FOR DATA AMOUNTS EXPERIMENT
        config['general']['dataset_amounts_experiment'] = False
        config['general']['trialNumber'] = endpoint

        expHandler = experimentHandler(config)
        expHandler.run_experiment(config)


        # # TEST ENSEMBLE CODE
        from src.evaluation.validate_on_test_set import validate_models_on_test_set

        # # # run the models on the test set
        #trial_dir = config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
        validate_models_on_test_set(config, trial_dir)
