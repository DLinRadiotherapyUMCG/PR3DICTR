import os

from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.experiments.experimentHandler import experimentHandler
from src.evaluation.validate_on_test_set import validate_models_on_test_set


if __name__ == '__main__':

    # Setup
    log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    # config = get_config('Xerostomia_model')
    config = get_config('HNC_OS_model')  # Load the config for the HNC OS model

    # Disable randomness
    set_random_seed(config['general']['seed'])

    config['general']['experiment_name'] = "OS_model_test_HP1"
    config['training']['max_epochs'] = 4
    #config['data']['kFolds']['n_iterations'] = 5
    

    # # MAIN: DL running class (with optional hyperparameter optimization)
    # this line is used to run the k-fold cross-validation
    expHandler = experimentHandler(config)
    expHandler.run_experiment(config)


    # TEST THE ENSEMBLE OF MODELS ON THE TEST SET
    
    # # # # run the models on the test set
    trial_dir = os.path.join(config['paths']['results'], config['general']['experiment_name'], config['general']['trialNumber'])
    validate_models_on_test_set(config, trial_dir)

