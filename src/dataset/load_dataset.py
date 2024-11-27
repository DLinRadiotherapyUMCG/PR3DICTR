import os
import logging
from typing import Optional, List, Tuple

from torch.utils.data import Dataset

from src.utils.data_equalizer import get_delimiter, get_umcg_n, data_split, label_equalizer
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def removePtnsExcluded(df, config):
    excludedVariables = config['data']['ExcludedVar']
    excludedValues = config['data']['ExcludedValue']

    if(len(excludedVariables) != len(excludedValues)):
        raise Exception("Exception: Exclusion parameters for patient removal need to have the same size. This contains the excluded variables and values.")
    
    for i in range(len(excludedVariables)):
        df = df.loc[df[excludedVariables[i]] != excludedValues[i]]

    return df






def ValidateImageDataExists(config, df):
    imagePath = config['paths']['images']
    ptnDirectories = os.listdir(imagePath)

    ptnClinList = df['PatientID'].tolist()
    removePtnIDS = []
    for i in range(len(ptnClinList)):
        zerosPtnNmbr = str(ptnClinList[i]).rjust(config['data']['patientID_length'],'0')
        if(ptnClinList[i] in ptnDirectories or zerosPtnNmbr in ptnDirectories):
            pass
        else:
            # Not found --> remove
            removePtnIDS.append(ptnClinList[i])
    
    print(f"Removed ptns = {len(removePtnIDS)}")
    df = df[(df['PatientID'].isin(removePtnIDS)) == False]
    return df

def load_dataset_single(csvPath, config, patient_ids = None):
    delimiterFound = get_delimiter(csvPath)
    dlDf = pd.read_csv(csvPath, delimiter=delimiterFound, dtype={'PatientID': str})

    return dlDf



def load_dataset(config, patient_ids=None):
    """
    
    """

    patientID_col = config['data']['patientVar']
    # load the dataset (patients are split by 'train_val'/'test') --> one dataframe
    dataset_csv_dir = os.path.join(config['paths']['csv'], config['data']['filename_stratified_sampling_test_csv'])

    delimiterFound = get_delimiter(dataset_csv_dir)
    df_total = pd.read_csv(dataset_csv_dir, delimiter=delimiterFound, dtype={'PatientID': str})
    # make sure the patientID strings are long enough
    patientID_length = config['data']['patientID_length']
    df_total[patientID_col] = df_total[patientID_col].apply(lambda x: x.rjust(patientID_length, '0'))  # ['%0.{}d'.format(patient_id_length) % int(x) for x in df[patient_id_col]]
    
    df_total = removePtnsExcluded(df_total, config)
    df_total = ValidateImageDataExists(config, df_total) 

    # TODO: Daniel: idk what the purpose of this was in the old code? was it ever used?
    # if patient_ids:
    #     totalDf = totalDf[totalDf[patientID_col].isin(patient_ids)]

    # if in test mode, and we want to use a subset of the data, then subsample the total dataset
    if config['general']['testMode'] and "n_patients_total" in config['data']:
        num_patients_sample = config['data']['n_patients_total']
        df_total = subsample_dataset(num_patients_sample, df_total)
    
    # split off the test set
    df_test = df_total[df_total[config['data']['splitvar']] == "test"]
    df_train_val = df_total[df_total[config['data']['splitvar']] != "test"]

    
    assert not PtnID_SanityCheck(config,df_train_val,df_test)

    return df_train_val, df_test




def generate_K_fold_cross_validation_splits(config, df_development_set):
    """
    Generates K train-val splits of the develoment dataset, using stratified K-Fold cross-validation.
    Returns a list of dictionaries, where each dictionary contains a train and a validation dataframe: [{'train': df_t1, 'val': df_v1}, ...]
    """
    # Check and validate if KFolds settings are active
    k_fold_dataframes_collection = []


    labels = df_development_set[config['columns']['label']]
    # encode the labels (makes it possible to use StratifiedKFold for multi-label problems, as it only works on binary or multi-class)
    encoded_labels = LabelEncoder().fit_transform([''.join(str(l)) for l in labels.values])
        
    skf = StratifiedKFold(n_splits=config["data"]["kFolds"]["Splits"], shuffle=True, random_state=config["general"]["seed"])
    for i, (train_index, val_index) in enumerate(skf.split(df_development_set,encoded_labels)):
        train_i_df = df_development_set.iloc[train_index]
        #trainDf_sel = mergeDf.iloc[train_index]
        if(config['data']['equalizer']['isEnabled']):
            train_i_df = label_equalizer(train_i_df, config)
        val_i_df = df_development_set.iloc[val_index]

        assert not PtnID_SanityCheck(config, train_i_df, val_i_df)

        k_fold_dataframes_collection.append({"train": train_i_df, "val": val_i_df})

    return k_fold_dataframes_collection



def generate_single_train_val_split(config, df_development_set):
    """
    A function to split the development set into 1 train and 1 validation set (in case you want to train just one model)
    # TODO: check that this works properly
    """
    labels = df_development_set[config['columns']['label']]
    train_df, val_df = train_test_split(df_development_set, test_size=config['data']['kFolds']['validation_size'], stratify=labels, random_state=config['general']['seed'])
    
    assert not PtnID_SanityCheck(config, train_df, val_df)
    
    return train_df, val_df







































def load_dataset_total(config, patient_ids = None):    # TODO: re-name to 'load_dataframes' ?
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
        totalDf = removePtnsExcluded(totalDf, config)
        totalDf = ValidateImageDataExists(config, totalDf)
        
        if patient_ids:
            totalDf = totalDf[totalDf['PatientID'].isin(patient_ids)]

        if(config['data']['splitvar'] != ""):
            splitVar = config['data']['splitvar']
            trainDf = totalDf[totalDf[splitVar] == "Train"]
            valDf = totalDf[totalDf[splitVar] == "Val"] 
            testDf = totalDf[totalDf[splitVar] == "Test"] 

            # BUG: This is how Daniel defined the splits
            if len(trainDf) == 0 and len(valDf) == 0:
                trainDf = totalDf[totalDf[splitVar] == "train_val"]
                testDf = totalDf[totalDf[splitVar] == "test"] 

                print("length test set", len(testDf))


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
    #print(f"Patient collection --> Train: {trainDf.shape[0]}, Validation: {valDf.shape[0]}")
    #if(testDf.shape[0] != 0):all you ne
    #    print(f"Patient collection --> Test: {testDf.shape[0]}")


    # if in test mode, and we want to use a subset of the data, then subsample the total dataset
    if config['general']['testMode'] and "n_patients_total" in config['data']:
        num_patients_sample = config['data']['n_patients_total']
        trainDf, valDf, testDf = subsample_datasets(num_patients_sample, trainDf, valDf, testDf)
        #print(len(trainDf), len(valDf), len(testDf))
    
    # Check and validate if KFolds settings are active
    trainDataset_Collection = []
    valDataset_Collection = []
    testDataset_Collection = []
    if(config["data"]["kFolds"]["isEnabled"] and config["data"]["kFolds"]["Iterations"] <= config["data"]["kFolds"]["Splits"]):
        # Multiple training and val datasets
        mergeDf = pd.concat([trainDf,valDf])
        labels = mergeDf[config['columns']['label']]
        # encode the labels (makes it possible to use StratifiedKFold for multi-label problems, as it only works on binary or multi-class)
        encoded_labels = LabelEncoder().fit_transform([''.join(str(l)) for l in labels.values])
         
        skf = StratifiedKFold(n_splits=config["data"]["kFolds"]["Splits"], shuffle=True, random_state=config["general"]["seed"])
        for i, (train_index, val_index) in enumerate(skf.split(mergeDf,encoded_labels)):
            trainDf_sel = mergeDf.iloc[train_index]
            if(config['data']['equalizer']['isEnabled']):
                trainDf_sel = label_equalizer(trainDf_sel, config)
            valDf_sel = mergeDf.iloc[val_index]

            # Check sanity is correct
            if(Complete_SanityCheck(config,[trainDf_sel,valDf_sel,testDf])):
                raise Exception("ABORT: Datasets contain identical patients! NOT ALLOWED!")

            trainDataset_Collection.append(trainDf_sel)
            valDataset_Collection.append(valDf_sel)
            testDataset_Collection.append(testDf)


            if(i == config["data"]["kFolds"]["Iterations"] - 1):
                break
    else:   
        # Single train and val dataset
        if(config['data']['equalizer']['isEnabled']):
                trainDf = label_equalizer(trainDf, config)
        if(Complete_SanityCheck(config,[trainDf,valDf,testDf])):
                raise Exception("ABORT: Datasets contain identical patients! NOT ALLOWED!")
        
        if(config['general']['testMode'] and trainDf.shape[0] > 100):
            # Only use 100 patients for training dataset
            trainDf = trainDf.iloc[:100]

        trainDataset_Collection.append(trainDf)
        valDataset_Collection.append(valDf)
        testDataset_Collection.append(testDf)
    

    logging.info(f"Patient amount in datasets: Train = {trainDataset_Collection[0].shape[0]}, Validation = {valDataset_Collection[0].shape[0]}, Test = {testDataset_Collection[0].shape[0]}")

    return [trainDataset_Collection, valDataset_Collection, testDataset_Collection]#, metadata






def subsample_dataset(num_patients_sample, df_dataset):
    """
    Subsample the datasets to that `num_patients_sample` are used in total.
    This is used for code-testing mode.
    """
    # find the number of patients in each dataset, so that we can preserve the ratio of patients in each dataset (train, val, test)
    n_total_loaded = df_dataset.shape[0]
    df_dataset = df_dataset.sample(frac=1, random_state=42).reset_index(drop=True) # shuffles the rows of the dataframe

    # Only use 100 patients for training dataset
    df_dataset = df_dataset.iloc[:int(n_total_loaded * num_patients_sample)]
    
    return df_dataset


def subsample_datasets(num_patients_sample, trainDf, valDf, testDf):
    """
    Subsample the datasets to that `num_patients_sample` are used in total (across all three datasets).
    This is used for test mode.
    """
    # find the number of patients in each dataset, so that we can preserve the ratio of patients in each dataset (train, val, test)
    n_train_loaded, n_val_loaded, n_test_loaded = trainDf.shape[0], valDf.shape[0], testDf.shape[0]
    n_total_loaded = n_train_loaded + n_val_loaded + n_test_loaded

    # Only use 100 patients for training dataset
    trainDf = trainDf.iloc[:int(n_train_loaded/n_total_loaded * num_patients_sample)]
    valDf = valDf.iloc[:int(n_val_loaded/n_total_loaded * num_patients_sample)]
    testDf = testDf.iloc[:int(n_test_loaded/n_total_loaded * num_patients_sample)]

    return trainDf, valDf, testDf



def Complete_SanityCheck(config,dfArray):
    for i in range(len(dfArray) - 1):
        for j in range(i + 1,len(dfArray)):
            if(PtnID_SanityCheck(config,dfArray[i],dfArray[j])):
                return True
    return False           


def PtnID_SanityCheck(config,df1,df2):
    return any(df1[config['data']['patientVar']].isin(df2[config['data']['patientVar']]))
