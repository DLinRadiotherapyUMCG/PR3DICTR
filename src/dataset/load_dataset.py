import os
from typing import Optional, List, Tuple

from torch.utils.data import Dataset

from src.dataset.HNCDataset import HNCDataset, ToxDataset
from src.utils.data_equalizer import get_delimiter, get_umcg_n, data_split, label_equalizer
import pandas as pd

from sklearn.model_selection import StratifiedKFold

def load_dataset(config, csv_path, patient_ids = None, augment= False, split = False, train = True, splitVar = "Split"):
    """
    Loads data for a single csv file.
    :param csv_path:
    :param config:
    :param patient_ids:
    :return: PyTorch Dataset and metadata
    """    
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

def ValidateImageDataExists(config, df):
    imagePath = config['paths']['images']
    ptnDirectories = os.listdir(imagePath)

    ptnClinList = df['PatientID'].tolist()
    removePtnIDS = []
    for i in range(len(ptnClinList)):
        zerosPtnNmbr = str(ptnClinList[i]).rjust(7,'0')
        if(ptnClinList[i] in ptnDirectories or zerosPtnNmbr in ptnDirectories):
            pass
        else:
            # Not found --> remove
            removePtnIDS.append(ptnClinList[i])
    
    #print(f"Removed ptns = {len(removePtnIDS)}")
    df = df[(df['PatientID'].isin(removePtnIDS)) == False]
    return df

def load_dataset_single(csvPath, config, patient_ids = None):
    delimiterFound = get_delimiter(csvPath)
    dlDf = pd.read_csv(csvPath, delimiter=delimiterFound, dtype={'PatientID': str})
    toxDataset = ToxDataset(config,dlDf)

    # Get example patient to get data info
    example_input, _, _ = toxDataset[0]
    channels, depth, height, width = example_input.shape
    n_features = len(config['columns']['clinical_features'])

    metadata = {
        "channels": channels,
        "depth": depth,
        "height": height,
        "width": width,
        "n_features": n_features,
    }
    return (toxDataset,metadata)

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

    # Check if 1 single file is given (testData is not always available)
    testDf = pd.DataFrame()
    if(trainfile == valfile):
        # Single file to split
        delimiterFound = get_delimiter(trainfile)
        totalDf = pd.read_csv(trainfile, delimiter=delimiterFound, dtype={'PatientID': str})

        totalDf = ValidateImageDataExists(config, totalDf)

        
        if patient_ids:
            totalDf = totalDf[totalDf['PatientID'].isin(patient_ids)]

        if(config['data']['splitvar'] != ""):
            splitVar = config['data']['splitvar']
            trainDf = totalDf[totalDf[splitVar] == "Train"]
            valDf = totalDf[totalDf[splitVar] == "Val"] 
            testDf = totalDf[totalDf[splitVar] == "Test"] 

            #if(config['data']['equalizer']['isEnabled']):
            #    trainDf = label_equalizer(trainDf, config)
        else:    
            # Need to split manual
            trainDf,valDf,testDf = data_split(totalDf, config, split=[0.7,0.15,0.15])

    else:
        # Two seperate files that are allready split
        delimiterFound = get_delimiter(trainfile)
        trainDf = pd.read_csv(trainfile, delimiter=delimiterFound, dtype={'PatientID': str})
        delimiterFound = get_delimiter(valfile)
        valDf = pd.read_csv(valfile, delimiter=delimiterFound, dtype={'PatientID': str})

    # Write information about data
    print(f"Patient collection --> Train: {trainDf.shape[0]}, Validation: {valDf.shape[0]}")
    if(testDf.shape[0] != 0):
        print(f"Patient collection --> Test: {testDf.shape[0]}")

    # Check and validate if KFolds settings are active
    trainDataset_Collection = []
    valDataset_Collection = []
    testDataset_Collection = []
    if(config["data"]["kFolds"]["isEnabled"] and config["data"]["kFolds"]["Iterations"] < config["data"]["kFolds"]["Splits"]):
        # Multiple training and val datasets
        mergeDf = pd.concat([trainDf,valDf])
        label = mergeDf[config['columns']['label']]
        skf = StratifiedKFold(n_splits=config["data"]["kFolds"]["Splits"], shuffle=True, random_state=config["general"]["seed"])
        for i, (train_index, val_index) in enumerate(skf.split(mergeDf,label)):
            trainDf_sel = mergeDf.iloc[train_index]
            if(config['data']['equalizer']['isEnabled']):
                trainDf_sel = label_equalizer(trainDf_sel, config)
            valDf_sel = mergeDf.iloc[val_index]

            # Check sanity is correct
            if(Complete_SanityCheck(config,[trainDf_sel,valDf_sel,testDf])):
                raise Exception("ABORT: Datasets contain identical patients! NOT ALLOWED!")

            trainDataset_Collection.append(ToxDataset(config,trainDf_sel))
            valDataset_Collection.append(ToxDataset(config,valDf_sel))
            testDataset_Collection.append(ToxDataset(config,testDf))

            if(i == config["data"]["kFolds"]["Iterations"] - 1):
                break
    else:   
        # Single train and val dataset
        if(config['data']['equalizer']['isEnabled']):
                trainDf = label_equalizer(trainDf, config)
        if(Complete_SanityCheck(config,[trainDf,valDf,testDf])):
                raise Exception("ABORT: Datasets contain identical patients! NOT ALLOWED!")
        trainDataset_Collection.append(ToxDataset(config,trainDf))
        valDataset_Collection.append(ToxDataset(config,valDf))
        testDataset_Collection.append(ToxDataset(config,testDf))

    # Get example patient to get data info
    example_input, _, _ = trainDataset_Collection[0][0]
    channels, depth, height, width = example_input.shape
    n_features = len(config['columns']['clinical_features'])

    metadata = {
        "channels": channels,
        "depth": depth,
        "height": height,
        "width": width,
        "n_features": n_features,
    }

    return [trainDataset_Collection, valDataset_Collection, testDataset_Collection], metadata

def Complete_SanityCheck(config,dfArray):
    for i in range(len(dfArray) - 1):
        for j in range(i + 1,len(dfArray)):
            if(PtnID_SanityCheck(config,dfArray[i],dfArray[j])):
                return True
    return False           


def PtnID_SanityCheck(config,df1,df2):
    return any(df1[config['data']['patientVar']].isin(df2[config['data']['patientVar']]))
