import os

import numpy as np

from src.Visualization.plot_slices import plot_slices
from src.load_config import load_config
from src.load_dataset import load_dataset
from src.utils.parse_args import parse_args
from src.utils.set_random_seed import set_random_seed

def main():
    # Setup
    toxicity, log_level = parse_args()

    # Load the config
    config = load_config(toxicity)

    # Load the dataset
    print('Loading dataset')
    train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=True)
    train_data_no_augment, _ = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=False)

    slices = [0, 30, 45, 60, 75, 95]
    rows = []

    for i in range(5):
        inputs, clinical_features, targets = train_data[i]
        inputs_no_augment, clinical_features_no_augment, targets_no_augment = train_data_no_augment[i]

        rows.append({
            'Label': f'Patient: {i}\nAugmented',
            'CT': inputs[0],
            'RTDOSE': inputs[1],
            'RTSTRUCT': inputs[2],
        })

        rows.append({
            'Label': f'Not Augmented',
            'CT': inputs_no_augment[0],
            'RTDOSE': inputs_no_augment[1],
            'RTSTRUCT': inputs_no_augment[2],
        })


    fig, axs = plot_slices(rows, slices)

    fig.show()

if __name__ == '__main__':
    # Setup
    toxicity, log_level = parse_args()

    # Load the config
    config = load_config(toxicity)

    # Load the dataset
    print('Loading dataset')
    train_data, metadata = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=True)
    train_data_no_augment, _ = load_dataset(config, os.path.join(config['paths']['csv'], 'train_full.csv'), augment=False)

    slices = [0, 30, 45, 60, 75, 95]
    rows = []

    for i in range(5):
        inputs, clinical_features, targets = train_data[i]
        inputs_no_augment, clinical_features_no_augment, targets_no_augment = train_data_no_augment[i]

        rows.append({
            'Label': f'Patient: {i}\nAugmented',
            'CT': inputs[0],
            'RTDOSE': inputs[1],
            'RTSTRUCT': inputs[2],
        })

        rows.append({
            'Label': f'Not Augmented',
            'CT': inputs_no_augment[0],
            'RTDOSE': inputs_no_augment[1],
            'RTSTRUCT': inputs_no_augment[2],
        })


    fig, axs = plot_slices(rows, slices)

    fig.show()