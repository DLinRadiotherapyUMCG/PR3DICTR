import os
import logging
from typing import Optional, List, Tuple

from torch.utils.data import Dataset

from src.utils.data_equalizer import get_delimiter, get_umcg_n, data_split, label_equalizer
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
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



def check_image_data_exists(config, df):
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



def load_dataset(config : dict, patient_ids=None):
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
    df_total = check_image_data_exists(config, df_total) 

    # TODO: Daniel: idk what the purpose of this was in the old code? was it ever used?
    # if patient_ids:
    #     totalDf = totalDf[totalDf[patientID_col].isin(patient_ids)]

    # if in test mode, and we want to use a subset of the data, then subsample the total dataset
    if config['general']['testMode'] and "n_patients_total" in config['data']:
        print("HELLO \nSubsampling the dataset for testing purposes.")
        num_patients_sample = config['data']['n_patients_total']
        df_total = subsample_dataset(num_patients_sample, df_total)
        print(len(df_total))
    
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

    # if the config dictates only 1 n_split, then just do a single train-val split (i.e. no K-fold cross-validation)
    if config["data"]["kFolds"]["n_splits"] == 1:
        logging.warning("WARNING: K-Fold cross-validation is set to 1 split. This is equivalent to a single train-val split.")
        
        train_i_df, val_i_df = generate_single_train_val_split(config, df_development_set)
        k_fold_dataframes_collection.append({"train": train_i_df, "val": val_i_df})
        #return [generate_single_train_val_split(config, df_development_set)]
    
    else:
        if config["data"]["kFolds"]["split_strategy"] == 'stratified':
            # encode the labels (makes it possible to use StratifiedKFold for multi-label problems, as it only works on binary or multi-class)
            labels = df_development_set[config['data']['stratify_on']]  # the columns to stratify by
            encoded_labels = LabelEncoder().fit_transform([''.join(str(l)) for l in labels.values])   
            kf_splitter = StratifiedKFold(n_splits=config["data"]["kFolds"]["n_splits"], shuffle=True, random_state=config["general"]["seed"])

            k_fold_splits = kf_splitter.split(df_development_set, encoded_labels)
        elif config["data"]["kFolds"]["split_strategy"] == 'random':
            kf_splitter = KFold(n_splits=config["data"]["kFolds"]["n_splits"], shuffle=True, random_state=config["general"]["seed"])

            k_fold_splits = kf_splitter.split(df_development_set)
        else:
            raise Exception("Exception: K-fold split strategy not recognized. Please check the configuration file.")


        for i, (train_index, val_index) in enumerate(k_fold_splits):
            train_i_df = df_development_set.iloc[train_index]
            #trainDf_sel = mergeDf.iloc[train_index]
            if(config['data']['equalizer']['isEnabled']):
                train_i_df = label_equalizer(train_i_df, config)
            val_i_df = df_development_set.iloc[val_index]

            assert not PtnID_SanityCheck(config, train_i_df, val_i_df)

            k_fold_dataframes_collection.append({"train": train_i_df, "val": val_i_df})

    
    return k_fold_dataframes_collection



# TODO: an option for a single train-val split
def generate_single_train_val_split(config, df_development_set):
    """
    A function to split the development set into 1 train and 1 validation set (in case you want to train just one model)
    # TODO: check that this works properly
    """
    if config["data"]["kFolds"]["split_strategy"] == 'stratified':
        labels = df_development_set[config['columns']['labels']]
    else:
        labels = None
    train_df, val_df = train_test_split(df_development_set, test_size=config['data']['kFolds']['validation_size'], stratify=labels, random_state=config['general']['seed'])
    
    assert not PtnID_SanityCheck(config, train_df, val_df)
    
    return train_df, val_df







































def subsample_dataset(num_patients_sample, df_dataset):
    """
    Subsample the datasets to that `num_patients_sample` are used in total.
    This is used for code-testing mode.
    """
    # find the number of patients in each dataset, so that we can preserve the ratio of patients in each dataset (train, val, test)
    n_total_loaded = df_dataset.shape[0]
    df_dataset = df_dataset.sample(frac=1, random_state=42).reset_index(drop=True) # shuffles the rows of the dataframe

    # Only use `num_patients_sample` patients for training dataset
    df_dataset = df_dataset.iloc[:int(num_patients_sample)]
    
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
