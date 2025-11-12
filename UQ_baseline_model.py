from src.evaluation.validate_on_test_set import validate_models_on_test_set
from src.experiments.experimentHandler import experimentHandler
from src.config_presets.tools.get_config import get_config, update_config
from src.config_presets.tools.load_config import load_config

from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed
from uncertainty_main import load_model_config_for_uncertainty_experiment

import src.hyper_opt.WandB_functions as WandB_functions


import faulthandler, signal
faulthandler.register(signal.SIGUSR1)




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
    endpoint = "LRC"
    #endpoint = "OS"

    config = load_model_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

    # SETTINGS FOR DATA AMOUNTS EXPERIMENT
    config['general']['dataset_amounts_experiment'] = False
    config['data']['equalizer']['isEnabled'] = False

    config['uncertainty']['deep_ensemble']['n_models'] = 10
    # config['data']['n_training_patients_list'] = [50, 100, 150, 200]

    config['general']['experiment_name'] = "Baseline UQ Models" 
    config['general']['trialNumber'] = f"{endpoint}_baseline_models"

    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)

    trial_dir = config['general']['resultsCurrentDirectory']
    trial_dir = f"/home/macraedc/UQ_results/Baseline UQ Models/{config['general']['trialNumber']}"
    validate_models_on_test_set(config, trial_dir)


    # # 
    # dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    

    # for d_rate in dropout_rates:
    #     config['general']['experiment_name'] = "Small MC Dropout"
    #     #config['uncertainty']['MC_dropout']['dropout_p'] = d_rate
    #     config['uncertainty']['MC_dropout']['dropout_p'] = d_rate

    #     config['general']['trialNumber'] = f"{int(d_rate*100)}" 

    #     train_MC_dropout_model(config, UQ_method="MC_dropout")


    # # MODEL TRAINING
    # config['general']['trialNumber'] = endpoint
    # config['general']['experiment_name'] = "Small Deep Ensemble"
    # set_random_seed(config['general']['seed'])
    # train_deep_ensemble_models(config)


    # 
    # config['general']['trialNumber'] = f"{endpoint}_with_oversampling_adam" 
    #config['data']['equalizer']['isEnabled'] = True
   




    


