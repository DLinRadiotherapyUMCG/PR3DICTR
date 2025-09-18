from src.config_presets.tools.get_config import get_config, update_config
from src.config_presets.tools.load_config import load_config

from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

import src.hyper_opt.WandB_functions as WandB_functions
from uncertainty_main import load_model_config_for_uncertainty_experiment

import faulthandler, signal
faulthandler.register(signal.SIGUSR1)





if __name__ == '__main__':
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    print(torch.cuda.memory_summary())


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

    endpoints = ["Dysphagia_M06", "Xerostomia_M06", "OS", "LRC"]
    endpoints = ['Taste_M06']

    for endpoint in endpoints:
        config = load_model_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

        # SETTINGS FOR DATA AMOUNTS EXPERIMENT
        config['general']['dataset_amounts_experiment'] = False

        # TTA

        config['general']['experiment_name'] = "TTA" 
        config['general']['trialNumber'] = endpoint

        set_random_seed(config['general']['seed'])
        train_MC_dropout_model(config, UQ_method="TTA")

        # DEEP ENSEMBLE

        config['general']['experiment_name'] = "Deep Ensemble"
        config['general']['trialNumber'] = endpoint

        set_random_seed(config['general']['seed'])
        train_deep_ensemble_models(config)

        


