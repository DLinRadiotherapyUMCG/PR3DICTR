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
    CONFIG_NAME = 'Daniel/Baoqiang_OS'
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
    
    
    # ["Geslacht", "Leeftijd", "Dysphagia_W01_Grade0_1", "Dysphagia_W01_Grade2", "Dysphagia_W01_Grade3_4"]
    config['general']['trialNumber'] = "OS_TRP_higher_lr"  # Set the trial number for the experiment
    #config['columns']['clinical_features'] = []
    # config['general']['experiment_name'] = "Deep Ensemble"   
    # train_deep_ensemble_models(config)
    # evaluate_deep_ensemble_models(config)

    
    config['general']['experiment_name'] = "MC Dropout" 
    train_MC_dropout_model(config)
    collect_bayesian_forward_passes(config, UQ_method="MC_dropout") # UQ_method='TTA')
    
    
    config['general']['experiment_name'] = "TTA" 
    train_MC_dropout_model(config)
    collect_bayesian_forward_passes(config, UQ_method='TTA')
    """
    
    config['general']['experiment_name'] = "Deep Ensemble"   
    train_deep_ensemble_models(config)
    evaluate_deep_ensemble_models(config)



    config = get_config(CONFIG_NAME)
    config['general']['trialNumber'] = "LRC_Densenet"  # Set the trial number for the experiment

    config['data']['stratify_on'] = ["Xerostomia_M06"]  # Set the label for stratification
    config['columns']['labels'] = ["Xerostomia_M06"]  # Set the label for TTA
    config['columns']['clinical_features'] = ["Geslacht", "Leeftijd", "Xerostomia_W01_Helemaal_niet", "Xerostomia_W01_Een_beetje", "Xerostomia_W01_Nogal_Heel_erg"]


    config['general']['experiment_name'] = "MC Dropout" 
    train_MC_dropout_model(config)
    collect_bayesian_forward_passes(config, UQ_method="MC_dropout") # UQ_method='TTA')
    
    config['general']['experiment_name'] = "TTA" 
    train_MC_dropout_model(config)
    collect_bayesian_forward_passes(config, UQ_method='TTA')
    
    config['general']['experiment_name'] = "Deep Ensemble"   
    train_deep_ensemble_models(config)
    evaluate_deep_ensemble_models(config)

    """

    
    

