import os
import copy

from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.experiments.experimentHandler import experimentHandler
from src.evaluation.validate_on_test_set import validate_models_on_test_set

from src.experiments.single_endpoint_models import run_single_toxicity_models_experiment


from src.uncertainty.deep_ensemble import train_deep_ensemble_models, evaluate_deep_ensemble_models
from src.uncertainty.MC_dropout import train_MC_dropout_model, collect_bayesian_forward_passes


if __name__ == '__main__':

    # Setup
    log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    config = get_config('Daniel/MICCAI/Multi_tox')

    # Disable randomness
    set_random_seed(config['general']['seed'])


    # config['model']['model_name'] = 'densenet'
    

    
    # experiment_name = 'ST_models'
    # config['columns']['labels_types'] = ['Binary']
    # run_single_toxicity_models_experiment(config, experiment_name)

    # Train MC Dropout models
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    model_config = copy.deepcopy(config) # load_model_config_for_uncertainty_experiment(copy.deepcopy(config), endpoint_name=endpoint)
    config["data"]["kFolds"]["split_strategy"] = None
    model_config['general']['dataset_amounts_experiment'] = False # ensure that that whole dataset is used

    # MC Dropout
    for d_rate in dropout_rates:
        model_config['general']['experiment_name'] = "MT MC Dropout in DenseNet"
        #config['uncertainty']['MC_dropout']['dropout_p'] = d_rate
        model_config['uncertainty']['MC_dropout']['dropout_p'] = d_rate

        model_config['general']['trialNumber'] = f"{int(d_rate*100)}" 

        train_MC_dropout_model(model_config, UQ_method="MC_dropout")


    # # Train DE models
    # model_config = copy.deepcopy(config) # 
    # model_config['general']['dataset_amounts_experiment'] = False # ensure that that whole dataset is used

    # model_config['general']['experiment_name'] = "MT Deep Ensemble"
    # #model_config['general']['trialNumber'] = endpoint

    # set_random_seed(config['general']['seed'])
    # train_deep_ensemble_models(model_config)
    # """






    # """





    # # Load the config
    # config = get_config('Daniel/MICCAI/Multi_tox')

    # # Disable randomness
    # set_random_seed(config['general']['seed'])
    

    # """
    # #This function runs the an experiment using different combinations of endpoints removal experiment. It will retrain the model on the dataset with one of the inputs (CT, dose, segmentations, clinical features) removed.
    # # Usage: It can be called from a main.py file, as long as you give it a config. It will then use the experimentHandler class to run the experiment.
    # """

    # assert config['hyperparam_tuning']['optuna']['isEnabled'] == False, "This experiment is not compatible with hyperparameter tuning. Please set 'isEnabled' to False in the config file."

    # # set the experiment name
    # #config['general']['experiment_name'] = experiment_name

    
    # # these are all of the image keys mentioned in the config file (i.e. a list of what we need to loop through)
    # original_endpoints_list = config['columns']['labels']


    # main_features = ['Geslacht', 'Leeftijd', # "Chemotherapy",
    #             #'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
    #             ]

    # baseline_features_dict = {
    #     'Aspiration_M06' : ['Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg'],
    #     'Dysphagia_M06' : ['Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4'],
    #     'Sticky_M06' : ['Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg'],
    #     'Taste_M06' : ['Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg'],
    #     'Xerostomia_M06' : ['Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'], 
    # }

    # for endpoint in original_endpoints_list:

    #     #config['general']['experiment_name'] = "TRP_HigherRes2_BEST_ST_models"
    #     config['general']['trialNumber'] = endpoint

    #     config['columns']['labels'] = [endpoint]
    #     config['columns']['clinical_features'] = main_features + baseline_features_dict[endpoint]

    #     config['data']['stratify_on'] = [endpoint]
    #     config['training']['loss']['BCE']['pos_weight'] = [1]

    #     # Train MC Dropout models
    #     dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        
    #     model_config = copy.deepcopy(config) # load_model_config_for_uncertainty_experiment(copy.deepcopy(config), endpoint_name=endpoint)
    #     config["data"]["kFolds"]["split_strategy"] = None
    #     model_config['general']['dataset_amounts_experiment'] = False # ensure that that whole dataset is used

    #     # MC Dropout
    #     for d_rate in dropout_rates:
    #         model_config['general']['experiment_name'] = f"ST MC Dropout/{endpoint}"
    #         #config['uncertainty']['MC_dropout']['dropout_p'] = d_rate
    #         model_config['uncertainty']['MC_dropout']['dropout_p'] = d_rate

    #         model_config['general']['trialNumber'] = f"{int(d_rate*100)}" 

    #         train_MC_dropout_model(model_config, UQ_method="MC_dropout")



    # for endpoint in original_endpoints_list:

    #     #config['general']['experiment_name'] = "TRP_HigherRes2_BEST_ST_models"
    #     config['general']['trialNumber'] = endpoint

    #     config['columns']['labels'] = [endpoint]
    #     config['columns']['clinical_features'] = main_features + baseline_features_dict[endpoint]

    #     config['data']['stratify_on'] = [endpoint]
    #     config['training']['loss']['BCE']['pos_weight'] = [1]

    #     # Train DE models
    #     model_config = copy.deepcopy(config) # 
    #     model_config['general']['dataset_amounts_experiment'] = False # ensure that that whole dataset is used

    #     model_config['general']['experiment_name'] = f"ST Deep Ensemble/{endpoint}"
    #     model_config['general']['trialNumber'] = endpoint

    #     set_random_seed(config['general']['seed'])
    #     train_deep_ensemble_models(model_config)

