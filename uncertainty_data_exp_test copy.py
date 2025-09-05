from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

import src.hyper_opt.WandB_functions as WandB_functions


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
    """
    config['general']['dataset_amounts_experiment'] = True

    config['uncertainty']['deep_ensemble']['n_models'] = 5
    # config['data']['n_training_patients_list'] = [50, 100, 150, 200]

    endpoints = ["LRC"] # ["Dysphagia_M06"] # , "Xerostomia_M06", "OS", "LRC"]

    for endpoint in endpoints:
        config = load_modal_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

        config['general']['dataset_amounts_experiment'] = True
        config['data']['n_training_patients_list'] = [50] # , 100, 150] #  [100, 200, 300, 400, 500, 600, 700, 800] # 
        
        config['uncertainty']['deep_ensemble']['n_models'] = 5

        

        config['uncertainty']['MC_dropout']['dropout_p'] = 0.2

        config['general']['experiment_name'] = "DATA MC Dropout" 
        config['general']['trialNumber'] = endpoint

        set_random_seed(config['general']['seed'])
        train_MC_dropout_model(config, UQ_method="MC_dropout")
    """
        

    config['general']['dataset_amounts_experiment'] = False

    config['uncertainty']['deep_ensemble']['n_models'] = 5
    # config['data']['n_training_patients_list'] = [50, 100, 150, 200]

    endpoints = ['Xerostomia_M06'] # ["Dysphagia_M06"] # "Xerostomia_M06"]  # ["LRC"] # ["Dysphagia_M06"] # , "Xerostomia_M06", "OS", "LRC"]

    for endpoint in endpoints:
        config = load_modal_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

        #for d_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
            #config['uncertainty']['MC_dropout']['dropout_p'] = d_rate

        #config['general']['experiment_name'] = "Tune MC Dropout Adam" 
        config['general']['trialNumber'] = endpoint # f"{int(d_rate*100)}"
    

        #config['uncertainty']['MC_dropout']['dropout_p'] = d_rate

        config['general']['experiment_name'] = "TTA" 

        set_random_seed(config['general']['seed'])
        # train_MC_dropout_model(config, UQ_method="MC_dropout")

        config['paths']['results'] = '/home/macraedc/rt_pred_results/'
        config['general']['resultsCurrentDirectory'] = f"/home/macraedc/rt_pred_results/TTA/{config['general']['trialNumber']}/model_1/"

        config['uncertainty']['TTA']['n_forward_passes'] = 50

        post_hoc_collect_bayesian_forward_passes(config, UQ_method = "TTA")



