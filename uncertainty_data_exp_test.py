from src.experiments.experimentHandler import experimentHandler
from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

import src.hyper_opt.WandB_functions as WandB_functions
import os

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
    
    from uncertainty_main import load_modal_config_for_uncertainty_experiment
    
    # ["Geslacht", "Leeftijd", "Dysphagia_W01_Grade0_1", "Dysphagia_W01_Grade2", "Dysphagia_W01_Grade3_4"]
    
    # config['general']['dataset_amounts_experiment'] = False

    config['uncertainty']['deep_ensemble']['n_models'] = 10

    training_patients_dict = {
        "OS" : [50, 100, 150, 200], 
        "LRC" : [50, 100, 150],
        "Dysphagia_M06" : [100, 200, 300, 400, 500, 600, 700, 800],
        "Xerostomia_M06" : [100, 200, 300, 400, 500, 600, 700, 800]
    }

    endpoints = ["Dysphagia_M06" , "Xerostomia_M06", "OS", "LRC"]
    endpoints = ["OS", "LRC"]

    for endpoint in endpoints:
        config = load_modal_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

    
        config['general']['experiment_name'] = "Normal UQ Models"
        config['general']['trialNumber'] = endpoint

        expHandler = experimentHandler(config)
        expHandler.run_experiment(config)


        # # TEST ENSEMBLE CODE
        from src.evaluation.validate_on_test_set import validate_models_on_test_set

        # # # run the models on the test set
        #trial_dir = config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
        validate_models_on_test_set(config, trial_dir)

        
        # config['general']['dataset_amounts_experiment'] = False
        # config['data']['n_training_patients_list'] = training_patients_dict[endpoint] # , 100, 150] #  [100, 200, 300, 400, 500, 600, 700, 800] # 


        # config['general']['experiment_name'] = "TTA" 
        # config['general']['trialNumber'] = endpoint

        # set_random_seed(config['general']['seed'])
        # train_MC_dropout_model(config, UQ_method="TTA")


        # config['general']['experiment_name'] = "Deep Ensemble" 
        # config['general']['trialNumber'] = endpoint

        # set_random_seed(config['general']['seed'])
        # # train_MC_dropout_model(config, UQ_method="MC_dropout")

        # train_deep_ensemble_models(config)


        # set_random_seed(config['general']['seed'])
    


