import copy
import os

from src.utils.set_random_seed import set_random_seed
from src.experiments.experimentHandler import experimentHandler
from src.evaluation.validate_on_test_set import validate_models_on_test_set


def get_single_tox_feature_set(endpoint):

    endpoint_features_dict = {"Aspiration_M06" : ['Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg'],
                              "Dysphagia_M06" : ['Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4'],
                              "Sticky_M06" : ['Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg'],
                              "Taste_M06" : ['Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg'],
                              "Xerostomia_M06" : ['Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'],
                              }
    
    return endpoint_features_dict[endpoint]



def run_single_toxicity_models_experiment(config, experiment_name, original_endpoints_list=None):
    """
    This function runs the an experiment using different combinations of endpoints removal experiment. It will retrain the model on the dataset with one of the inputs (CT, dose, segmentations, clinical features) removed.
    Usage: It can be called from a main.py file, as long as you give it a config. It will then use the experimentHandler class to run the experiment.
    """

    assert config['hyperparam_tuning']['optuna']['isEnabled'] == False, "This experiment is not compatible with hyperparameter tuning. Please set 'isEnabled' to False in the config file."

    # set the experiment name
    config['general']['experiment_name'] = experiment_name

    
    # these are all of the image keys mentioned in the config file (i.e. a list of what we need to loop through)
    if original_endpoints_list == None:
        original_endpoints_list = config['columns']['labels']


    main_features = ['Geslacht', 'Leeftijd', # "Chemotherapy",
                #'Loctum2_Larynx', 'Loctum2_Oral_Cavity', 'Loctum2_Overig', 'Loctum2_Pharynx',
                ]

    baseline_features_dict = {
        'Aspiration_M06' : ['Aspiration_W01_Helemaal_niet','Aspiration_W01_Een_beetje',  'Aspiration_W01_Nogal_Heel_erg'],
        'Dysphagia_M06' : ['Dysphagia_W01_Grade0_1', 'Dysphagia_W01_Grade2', 'Dysphagia_W01_Grade3_4'],
        'Sticky_M06' : ['Sticky_W01_Helemaal_niet', 'Sticky_W01_Een_beetje', 'Sticky_W01_Nogal_Heel_erg'],
        'Taste_M06' : ['Taste_W01_Helemaal_niet','Taste_W01_Een_beetje',  'Taste_W01_Nogal_Heel_erg'],
        'Xerostomia_M06' : ['Xerostomia_W01_Helemaal_niet', 'Xerostomia_W01_Een_beetje', 'Xerostomia_W01_Nogal_Heel_erg'], 
    }

    for endpoint in original_endpoints_list:
        experiment_config = copy.deepcopy(config)
        set_random_seed(experiment_config['general']['seed'])

        #config['general']['experiment_name'] = "TRP_HigherRes2_BEST_ST_models"
        config['general']['trialNumber'] = endpoint

        config['columns']['labels'] = [endpoint]
        config['columns']['clinical_features'] = main_features + baseline_features_dict[endpoint]

        config['data']['stratify_on'] = [endpoint]
        config['training']['loss']['BCE']['pos_weight'] = [1]

        # train 5-fold model
        expHandler = experimentHandler(config)
        expHandler.run_experiment(config)



        # TEST ENSEMBLE CODE

        #trial_dir = r"/scratch/s3719332/rt_pred_results/HP_TRP_HigherRes2_BEST/Trial_32_SGD1_GN_101/" # config['general']['resultsCurrentDirectory']
        trial_dir = os.path.join(config["paths"]["results"], config['general']['experiment_name'], config['general']['trialNumber'])
        # # run the models on the test set
        validate_models_on_test_set(config, trial_dir)
