import os
from typing import Optional, List, Tuple

from torch.utils.data import Dataset

from src.HNCDataset import HNCDataset, ToxDataset
from src.HNCDataset import get_delimiter, get_umcg_n, Data_split
import pandas as pd

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

def load_dataset_total(config, patient_ids = None):
    """
    Load the all datasets with handling the options in
    the config file. This includes the csvFile
    """
    # Get paths of the files
    trainfile = os.path.join(config['paths']['csv'], config['data']['trainfile'])
    valfile = os.path.join(config['paths']['csv'], config['data']['valfile'])

    # There are 3 options to get the train, validation data
    #   1) There are 2 seperate files
    #   2) Single file that contains splitVar
    #   3) Single file with no splitVar will custom be split in --> train,var,test (warning: test need to be unique by seed --> need to check)

    # Check if 1 single file is given
    if(trainfile == valfile):
        # Single file to split
        delimiterFound = get_delimiter(trainfile)
        totalDf = pd.read_csv(trainfile, delimiter=delimiterFound, dtype={'PatientID': str})
        
        if patient_ids:
            totalDf = totalDf[totalDf['PatientID'].isin(patient_ids)]

        if(config['data']['splitvar'] != ""):
            splitVar = config['data']['splitvar']
            trainDf = totalDf[totalDf[splitVar] == "train"]
            valDf = totalDf[totalDf[splitVar] == "val"] 
        else:    
            # Need to split manual
            trainDf,valDf,testDf = Data_split(totalDf, config, split=[0.7,0.15,0.15])

    else:
        # Two seperate files that are allready split
        delimiterFound = get_delimiter(trainfile)
        trainDf = pd.read_csv(trainfile, delimiter=delimiterFound, dtype={'PatientID': str})
        delimiterFound = get_delimiter(valfile)
        valDf = pd.read_csv(valfile, delimiter=delimiterFound, dtype={'PatientID': str})

    trainDataset = ToxDataset(config,trainDf)
    valDataset = ToxDataset(config,valDf)

    example_input, _, _ = trainDataset[0]
    channels, depth, height, width = example_input.shape

    n_features = len(config['columns']['clinical_features'])

    metadata = {
        "channels": channels,
        "depth": depth,
        "height": height,
        "width": width,
        "n_features": n_features,
    }

    return [trainDataset,valDataset], metadata

