import logging
import os

#import wandb
from src.get_config import get_config
from src.load_dataset import load_dataset
from src.save_model import save_model
from src.train_multi import train
from src.utils.logging import setup_logging
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed


# def main():
#     # Setup
#     toxicity, log_level = parse_args()
#     setup_logging(log_level)
#     wandb.login()
#     wandb.init(project=toxicity, job_type='train')

#     # Load the config
#     config = load_config(toxicity)

#     # Disable randomness
#     set_random_seed(config['seed'])

#     # Load the dataset
#     logging.info('Loading dataset')
#     train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=True)
#     val_data, _ = load_dataset(config, os.path.join(config['paths']['csv'], 'valid.csv'))

#     # Start training
#     model = train(config, train_data, val_data, metadata)

#     # Save the model
#     save_model(config, model, 'dl_model_full.pth')

if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()
    setup_logging(log_level)
    # wandb.login()
    # wandb.init(project=toxicity, job_type='train')
    # Load the config
    config = get_config('Multi_time')

    # Disable randomness
    set_random_seed(config['seed'])

    # Load the dataset
    logging.info('Loading dataset')
    train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=True)
    val_data, _ = load_dataset(config, os.path.join(config['paths']['csv'], 'valid.csv'))

    # Start training
    
    model = train(config, train_data, val_data, metadata)

    # Save the model
    save_model(config, model, 'dl_model_full.pth')


