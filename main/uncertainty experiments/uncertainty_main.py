from src.config_presets.tools.get_config import get_config, update_config
from src.config_presets.tools.load_config import load_config

from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

import src.hyper_opt.WandB_functions as WandB_functions

import faulthandler, signal
faulthandler.register(signal.SIGUSR1)


def load_model_config_for_uncertainty_experiment(config, endpoint_name = "Dysphagia_M06"):
    if endpoint_name == "OS":
        model_config = load_config("Daniel/uncertainty_models/Baoqiang_OS")
        
    elif endpoint_name == "LRC":
        model_config = load_config("Daniel/uncertainty_models/Baoqiang_LRC")
    elif endpoint_name == "Dysphagia_M06":
        model_config = load_config("Daniel/uncertainty_models/Suzanne_Dysphagia")
    elif endpoint_name == "Xerostomia_M06":
        model_config = load_config("Daniel/uncertainty_models/Hung_Xerostomia")
    elif endpoint_name == "Taste_M06":
        model_config = load_config("Daniel/uncertainty_models/Taste")
    elif endpoint_name == "Sticky_M06":
        model_config = load_config("Daniel/uncertainty_models/Sticky")
    elif endpoint_name == "Metal_artefact":
        model_config = load_config("Daniel/uncertainty_models/Metal_artefact")
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
    endpoint = "Xerostomia_M06"
    #endpoint = "OS"

    config = load_model_config_for_uncertainty_experiment(config, endpoint_name=endpoint)

    # SETTINGS FOR DATA AMOUNTS EXPERIMENT
    config['general']['dataset_amounts_experiment'] = False
    config['data']['equalizer']['isEnabled'] = False

    config['uncertainty']['deep_ensemble']['n_models'] = 10
    # config['data']['n_training_patients_list'] = [50, 100, 150, 200]

    config['general']['experiment_name'] = "MC Dropout no Pharynx in training" 
    config['general']['trialNumber'] = endpoint

    #config['columns']['clinical_features'] = []                      # NOTE: DANIEL CHANGED THIS

    config['data']['dataset_csv'] = "UQ_PRIMA_2025_no_pharynx_in_development_set.csv"


    train_MC_dropout_model(config, UQ_method="MC_dropout")


    # # config['general']['experiment_name'] = "Tune MC Sticky" 
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
   




    


