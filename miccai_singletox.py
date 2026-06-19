import os

from src.config_presets.tools.get_config import get_config
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

from src.experiments.experimentHandler import experimentHandler
from src.evaluation.validate_on_test_set import validate_models_on_test_set

from src.experiments.single_endpoint_models import run_single_toxicity_models_experiment

if __name__ == '__main__':

    # Setup
    log_level = parse_args()
    setup_logging(log_level)

    # Load the config
    config = get_config('Daniel/MICCAI/Multi_tox')

    # Disable randomness
    set_random_seed(config['general']['seed'])
    

    experiment_name = 'ST_models'
    original_endpoints_list = ['Dysphagia_M06', 'Aspiration_M06', 'Sticky_M06']
    # original_endpoints_list = ['Xerostomia_M06', 'Taste_M06']

    run_single_toxicity_models_experiment(config, experiment_name, original_endpoints_list)