import logging
import os
import uuid

import wandb

from src.constants import ENSEMBLE_MEMBERS
from src.config_presets.tools.load_config import load_config
from src.dataset.load_dataset import load_dataset
from src.models.tools.save_model import save_model
from src.training.train import train
from src.utils.logging.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

# Generate unique tag for the ensemble, before setting random seed
ensemble_run_tag = str(uuid.uuid4())
ensemble_run_tag = ensemble_run_tag[len(ensemble_run_tag) - 8:]

def main():
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)
    wandb.login()

    # Load the config
    config = load_config(toxicity)

    # Disable randomness
    set_random_seed(config['seed'])

    # Loop over ensembles
    for model_index in range(ENSEMBLE_MEMBERS):
        # Start a new wandb run for each ensemble member
        wandb.init(project=f'{toxicity}-Ensemble', job_type='train', tags=[ensemble_run_tag])

        # Load the dataset
        logging.info(f'Loading dataset for model {model_index}')
        train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], f'train_{model_index}.csv'))
        val_data, _ = load_dataset(config, os.path.join(config['paths']['csv'], f'valid.csv'))

        # Start training
        logging.info(f'Starting training for model {model_index}')
        model = train(config, train_data, val_data, metadata)

        # Stop the wandb run
        wandb.finish()

        # Save the model
        save_model(config, model, f'dl_model_{model_index}.pth')


if __name__ == '__main__':
    main()
