import logging
import os
import yaml
import pandas as pd
import torch
from src.utils.fileHandler import create_folder, create_file

def save_model(config, model, modelFile):
    """
    Save the model to the output directory.
    :param config:
    :param model:
    :param model_path:
    :return:
    """
    pathToSave = config['general']['resultsCurrentDirectory']
    fileLocation = os.path.join(pathToSave, modelFile)

    # Create folder if does not exist
    create_folder(pathToSave)

    # Log and Save
    logging.info(f'Saving model to {fileLocation}')
    torch.save(model.state_dict(), fileLocation)



def load_model(config, model, modelFile):
    """
    Load the model from the output directory.
    :param config:
    :param model:
    :param model_path:
    :return:
    """
    pathToSave = config['general']['resultsCurrentDirectory']
    fileLocation = os.path.join(pathToSave, modelFile)

    # Log and Save
    logging.info(f'Loading model from {fileLocation}')
    model.load_state_dict(torch.load(fileLocation))
    return model


def save_dataset(config, dataset, fileName):
    """
    Save the dataset to the output directory
    - dataset --> Pytorch dataset object
    """
    #print("Dataset")
    df = dataset
    #print(df)
    if(df.shape[0] == 0):
        return

    # The dataset will be saved for the given internal dataframe
    if(fileName.endswith(".csv") == False):
        fileName = f"{fileName}.csv"
    #print(fileName)
    directoryToSave = config['general']['resultsCurrentDirectory']
    pathToSave = os.path.join(directoryToSave,fileName)
    #print(pathToSave)
    df.to_csv(pathToSave, sep=";")


def save_config(config, fileName):
    directoryToSave = config['general']['resultsCurrentDirectory']
    if(fileName.endswith(".yaml") == False):
        fileName = f"{fileName}.yaml"
    pathConfig = os.path.join(directoryToSave,fileName)
    with open(pathConfig,'w') as outfile:
        yaml.dump(config,outfile,default_flow_style=False)


def save_dataset_summary(config, trainDataset, valDataset, testDataset):
    # Get the patientIds from the different datasets
    ptnVar = config['data']['patientVar']
    trainDataset_ptns = trainDataset[ptnVar].tolist()
    valDataset_ptns = valDataset[ptnVar].tolist()
    testDataset_ptns = []
    if(trainDataset.shape[0] != 0):
        testDataset_ptns = testDataset[ptnVar].tolist()
    
    # Merge the dataframes to a single again
    dataframes = [trainDataset,valDataset,testDataset]
    dfMerged = pd.concat(dataframes)

    # Select the columns used during the analysis
    #dfMerged = dfMerged[[config['columns']['label']] + config['columns']['clinical_features']]

    # Create column in which dataset it was found
    split = []
    for i in range(dfMerged.shape[0]):        
        if(dfMerged.iloc[i][ptnVar] in trainDataset_ptns):
            split.append("Train")
        elif(dfMerged.iloc[i][ptnVar] in valDataset_ptns):
            split.append("Val")
        elif(dfMerged.iloc[i][ptnVar] in testDataset_ptns):
            split.append("Test")
        else:
            split.append("Unknown")

    # Add new column
    dfMerged["split"] = split

    # Save data
    directoryToSave = config['general']['resultsCurrentDirectory']
    pathToSave =os.path.join(directoryToSave, "Dataset_Summary.csv")
    dfMerged.to_csv(pathToSave, sep=";")