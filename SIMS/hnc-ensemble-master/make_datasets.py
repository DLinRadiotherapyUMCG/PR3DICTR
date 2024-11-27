import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.constants import ENSEMBLE_MEMBERS, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT

# Paths
dataset_path = 'private/dataset/'
patients_per_toxicity_path = 'per_toxicity_csv_542/'
output_path = 'private/bagged_datasets/'

# Disable randomness
seed = 42
np.random.seed(seed)

toxicities = [
    {
        'name': 'xerostomia_m06',
        'file': 'Xerostomia_M06_542'
    },
    {
        'name': 'aspiration_m06',
        'file': 'Aspiration_M06_542'
    },
    {
        'name': 'dysphagia_m06',
        'file': 'Dysphagia_M06_542'
    },
    {
        'name': 'sticky_m06',
        'file': 'Sticky_M06_542'
    },
    {
        'name': 'taste_m06',
        'file': 'Taste_M06_542'
    }
]

for toxicity in toxicities:
    dataset_all_features_path = dataset_path + patients_per_toxicity_path + toxicity['file'] + '.csv'
    bagged_datasets_path = output_path + toxicity['name'] + '/'

    # Make the directory if it does not exist
    if not os.path.exists(bagged_datasets_path):
        os.makedirs(bagged_datasets_path)

    # Load all patients
    patients_all_features = pd.read_csv(dataset_all_features_path)

    # Calculate the test set size
    test_size = TEST_SPLIT / (TEST_SPLIT + VALID_SPLIT)

    # Split the dataset into training and temporary sets
    train, temp = train_test_split(patients_all_features, test_size=1 - TRAIN_SPLIT, random_state=seed)

    # Split the temporary set into validation and test sets
    valid, test = train_test_split(temp, test_size=test_size, random_state=seed)

    # Number of boostrap sample sets and samples per set
    num_bootstrap_sets = ENSEMBLE_MEMBERS
    num_samples_per_set = len(train)

    # Store sample sets
    bootstrap_sample_sets = []

    # Create bootstrap sample sets
    for i in range(num_bootstrap_sets):
        bootstrap_sample = train.sample(n=num_samples_per_set, replace=True, random_state=seed + i)
        bootstrap_sample_sets.append(bootstrap_sample)

    # Save the validation and testing datasets
    valid.to_csv(bagged_datasets_path + 'valid.csv', index=False)
    test.to_csv(bagged_datasets_path + 'test.csv', index=False)
    train.to_csv(bagged_datasets_path + 'train_full.csv', index=False)

    # Save the bagged datasets
    for i in range(num_bootstrap_sets):
        bootstrap_sample_sets[i].to_csv(bagged_datasets_path + 'train_' + str(i) + '.csv', index=False)


