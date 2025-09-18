import os
import logging
from src.utils.data_equalizer import get_delimiter, label_equalizer
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.dataset.LabelTypesManager import LabelTypesManager

from src.constants import PATIENT_ID_LENGTHS_DICT, PATIENT_ID_COL_NAME, SPLIT_COL_NAME


def remove_excluded_patients(df : pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Function to remove patients that should be excluded. This filtering is performed on certain columns, within which patients with certain values are dropped.
    (e.g. column='Chemotherapy' and value='True' to exclude patients who have recieved chemotherapy)
    Args:
        df (pd.DataFrame): dataset dataframe
        config (dict): config parameters
    Returns
        df (pd.Dataframe): dataset dataframe with excluded patients removed
    """
    excludedVariables = config['data']['excluded_variable_name']  # column name
    excludedValues = config['data']['excluded_values']           # values

    if len(excludedVariables) != len(excludedValues):
        raise Exception("Exception: Exclusion parameters for patient removal need to have the same size. This contains the excluded variables and values.")
    
    # for each column name, remove patients with the excluded value
    for i in range(len(excludedVariables)):
        df = df.loc[df[excludedVariables[i]] != excludedValues[i]]

    # return the filtered dataframe (i.e. the dataset without the excluded patients)
    return df




def check_image_data_exists(config: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes patients from the dataframe who are missing image data files.
    Args:
        config (dict): config parameters
        df (pd.DataFrame): dataset dataframe
    Returns:
        pd.DataFrame: dataframe without patients missing image data
    """
    patientID_length = PATIENT_ID_LENGTHS_DICT[config['data']['source']]
    image_path = config['paths']['images']
    available_dirs = set(os.listdir(image_path))

    def has_image_data(patient_id):
        padded_id = str(patient_id).rjust(patientID_length, '0')
        return str(patient_id) in available_dirs or padded_id in available_dirs

    mask = df['PatientID'].apply(has_image_data)
    removed_count = (~mask).sum()
    logging.info(f"Removed patients (no image data) = {removed_count}")

    return df[mask].reset_index(drop=True)





def load_dataset(config : dict, modelCard = None):
    """
    Loads the entire dataset from one dataset_csv. Removes any patients that should be excluded or that are missing iamge data, subsamples the dataset (if needed),
    and returns a dataframe containing the train_val patients and a dataframe with the test patients.
    Args:
        config (dict): the config params
    Returns:
        df_train_val (pd.DataFrame): dataframe with the training and validation set patients
        df_test (pd.DataFrame): dataframe containing only the test set patients
    """

    patientID_col = PATIENT_ID_COL_NAME
    # load the dataset (patients are split by 'train_val'/'test') --> one dataframe
    dataset_csv_dir = os.path.join(config['paths']['csv'], config['data']['dataset_csv'])

    delimiterFound = get_delimiter(dataset_csv_dir)
    df_total = pd.read_csv(dataset_csv_dir, delimiter=delimiterFound, dtype={'PatientID': str})
    # make sure the patientID strings are long enough
    patientID_length = PATIENT_ID_LENGTHS_DICT[config['data']['source']]
    df_total[patientID_col] = df_total[patientID_col].apply(lambda x: x.rjust(patientID_length, '0'))  # ['%0.{}d'.format(patient_id_length) % int(x) for x in df[patient_id_col]]
    
    df_total = remove_excluded_patients(df_total, config)
    df_total = check_image_data_exists(config, df_total) 

    # if in test mode, and we want to use a subset of the data, then subsample the total dataset
    if config['general']['testMode'] and "n_patients_total" in config['data']:
        logging.info("Subsampling the dataset for testing purposes.")
        num_patients_sample = config['data']['n_patients_total']
        df_total = subsample_dataset(num_patients_sample, df_total)
        logging.info(f"Dataset now contains {len(df_total)} patients (after subsampling).")
    
    # split off the test set
    df_test = df_total[df_total[SPLIT_COL_NAME] == "test"]
    df_train_val = df_total[df_total[SPLIT_COL_NAME] != "test"]
    
    # Show dataset size
    train_size = df_train_val.shape[0]
    test_size = df_test.shape[0]
    total_size = df_total.shape[0]
    logging.info(f"Train/Val dataset {train_size} ({train_size/total_size*100}%), Test dataset {test_size} ({test_size/total_size*100}%)")
    if modelCard is not None:
        modelCard.insert(["training_data","number_of_patients"],total_size)

    # check patients not in same dataset
    assert not has_patient_overlap(config,df_train_val,df_test)

    return df_train_val, df_test




def generate_K_fold_cross_validation_splits(config : dict, df_development_set : pd.DataFrame) -> list:
    """
    Generates K train-val splits of the develoment dataset, using stratified K-Fold cross-validation.
    Returns a list of dictionaries, where each dictionary contains a train and a validation dataframe: [{'train': df_t1, 'val': df_v1}, ...]
    Args: 
        config (dict): config params
        df_development_set (pd.DataFrame): a dataframe of the development set (i.e. all patients in the train or validation sets) to use in K-fold cross validation
    Returns:
        k_fold_dataframes_collection (list): list of dictionaries. Each dictionary contains a 'train' and a 'val' dataframe
    """
    # Check and validate if KFolds settings are active
    k_fold_dataframes_collection = []

    # print("CONFIG N_SPLITS", config["data"]["kFolds"]["n_splits"])

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
            # select the training and validation patients
            train_i_df = df_development_set.iloc[train_index]
            val_i_df   = df_development_set.iloc[val_index]

            assert not has_patient_overlap(config, train_i_df, val_i_df)

            
            k_fold_dataframes_collection.append({"train": train_i_df, "val": val_i_df})

    
    return k_fold_dataframes_collection







def generate_single_train_val_split(config, df_development_set : pd.DataFrame):
    """
    A function to split the development set into 1 train and 1 validation set (in case you want to train just one model)
    Args: 
        config (dict): config params
        df_development_set (pd.DataFrame): a dataframe of the development set (i.e. all patients in the train or validation sets) to use in K-fold cross validation
    Returns:
        train_df (pd.DataFrame): dataframe containing the training patients
        val_df (pd.DataFrame): dataframe containing the validation patients
    """
    # if the train-val split is stratified or random
    if config["data"]["kFolds"]["split_strategy"] == 'stratified':
        LabelManager = LabelTypesManager(config)  # create a LabelTypesManager object to handle the labels
        label_columns = LabelManager.label_names_full_list  # get the full list of labels, (including event and days labels for event endpoints)

        # Flatten label_columns in case it contains tuples
        flattened_label_columns = []
        for col in label_columns:
            if isinstance(col, (list, tuple)):
                flattened_label_columns.append(col[0]) # here, we only need to stratify the event column (not the time column)
            else:
                flattened_label_columns.append(col)
        
        labels = df_development_set[flattened_label_columns]

    else:
        labels = None

    # perform the split
    train_df, val_df = train_test_split(df_development_set, test_size=config['data']['kFolds']['validation_size'], stratify=labels, random_state=config['general']['seed'])
    
    # ensure theres no overlap between the sets
    assert not has_patient_overlap(config, train_df, val_df)
    
    return train_df, val_df






def subsample_dataset(num_patients_sample, df_dataset : pd.DataFrame) -> pd.DataFrame:
    """
    Subsample the datasets to that `num_patients_sample` are used in total.
    This is used for code-testing mode.
    """
    # Shuffle the dataset, and then use the first `num_patients_sample` patients as the sample
    df_dataset = df_dataset.sample(frac=1, random_state=42).reset_index(drop=True) # shuffles the rows of the dataframe
    df_dataset = df_dataset.iloc[:int(num_patients_sample)]  # take the sample 
    
    return df_dataset



     


def has_patient_overlap(config, df1 : pd.DataFrame, df2 : pd.DataFrame):
    """
    Check if there are any patients in both dataframes.
    Args:
        config (dict): config params
        df1 (pd.DataFrame): first dataframe
        df2 (pd.DataFrame): second dataframe
    Returns:
        bool: True if there is any patient overlap, False otherwise
    """
    return any(df1[PATIENT_ID_COL_NAME].isin(df2[PATIENT_ID_COL_NAME]))
