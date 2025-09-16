from src.config_presets.tools.get_config import get_config, update_config
from src.config_presets.tools.load_config import load_config

from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

import src.hyper_opt.WandB_functions as WandB_functions

import faulthandler, signal
faulthandler.register(signal.SIGUSR1)

import os

from uncertainty_main import load_model_config_for_uncertainty_experiment

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
    

    from src.uncertainty.deep_ensemble import train_deep_ensemble_models, evaluate_deep_ensemble_models, external_evaluate_deep_ensemble_models
    from src.uncertainty.MC_dropout import train_MC_dropout_model, collect_bayesian_forward_passes, post_hoc_collect_bayesian_forward_passes
    
    # Choose the endpoint
    endpoint = "Xerostomia_M06"
    #endpoint = "OS"

    config = load_model_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

    config['data']['source'] = 'MDACC'
    config['data']['dataset_csv'] = "MDACC_test_set_no_unknown_primary.csv"
    config['columns']['clinical_features'] = [col_name.replace("W01", "BSL") for col_name in config['columns']['clinical_features']]  # force MDACC to use BSL (and not W01) features

    config['paths']['csv'] = r"/home/macraedc/data/"
    config['paths']['images'] = r"/home/macraedc/data/MDACC_dataset/patients"

    config['general']['experiment_name'] = f"Tune MC Xerostomia SGD"

    # for dropout_rate in ["10", "20", "30", "40", "50"]:

    #     config['general']['trialNumber'] = dropout_rate

    #     config['general']['resultsCurrentDirectory'] = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'], 'model_1')


    #     post_hoc_collect_bayesian_forward_passes(config, UQ_method="MC_dropout")

    # MODEL TRAINING

    # config['general']['experiment_name'] = "TTA"
    # config['general']['trialNumber'] = endpoint
    # config['general']['resultsCurrentDirectory'] = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'], 'model_1')
    # post_hoc_collect_bayesian_forward_passes(config, UQ_method="TTA")


    # DEEP ENSEMBLE
    config['general']['experiment_name'] = "Deep Ensemble"
    config['general']['trialNumber'] = endpoint
    config['general']['resultsCurrentDirectory'] = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    experiment_dir = config['general']['resultsCurrentDirectory']
    # external_evaluate_deep_ensemble_models(config, experiment_dir)

    evaluate_deep_ensemble_models(config, experiment_dir)
    


    # 
    # config['general']['trialNumber'] = f"{endpoint}_with_oversampling_adam" 
    #config['data']['equalizer']['isEnabled'] = True
   




    


