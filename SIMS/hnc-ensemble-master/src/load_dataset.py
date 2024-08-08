from typing import Optional, List, Tuple

from torch.utils.data import Dataset

from src.HNCDataset import HNCDataset

def load_dataset(config, csv_path, patient_ids = None, augment= False, split = False, train = True, splitVar = "Split"):
    """
    Loads data for a single csv file.
    :param csv_path:
    :param config:
    :param patient_ids:
    :return: PyTorch Dataset and metadata
    """
    print(csv_path)
    
    # Create an instance of the HNCDataset
    dataset = HNCDataset(csv_path, config, patient_ids, augment=augment, 
                         split = split, train = train, splitVar = splitVar)

    # Get an example input to determine the metadata
    example_input, _, _ = dataset[0]
    channels, depth, height, width = example_input.shape

    n_features = len(config['columns']['clinical_features'])

    metadata = {
        "channels": channels,
        "depth": depth,
        "height": height,
        "width": width,
        "n_features": n_features,
    }

    # Return the dataset and the metadata
    return dataset, metadata
