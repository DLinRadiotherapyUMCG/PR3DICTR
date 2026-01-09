import os
import json
import math
import torch
import random
import logging
import numpy as np
import pandas as pd
from monai.data import Dataset, CacheDataset, PersistentDataset, SmartCacheDataset
from multiprocessing import Manager

from src.dataset.LabelTypesManager import LabelTypesManager
from src.dataset.transforms.MixUp import MixUp
from src.dataset.PatientDataLoader import PatientDataLoader
from src.dataset.utils.collect_metadata import collect_metadata
from src.constants import PATIENT_ID_COL_NAME

def prepare_data_dictionaries(config: dict, df: pd.DataFrame):
    """
    Reformats a dataframe into a list of patient dictionaries for Dataset classes.
    Args:
        config (dict): configuration object
        df (pd.DataFrame): dataframe for this dataloader's dataset
    Returns:
        data_dicts (list of dict): one dict per patient
        patient_ids_list (list): patient IDs in this dataset
    """
    patients_data_dir = config['paths']['images'] # Directory containing any volumetric data per patient
    image_keys = config['data']['image_keys'] # Keys that indicate the different volumetric data 
    clinical_features_columns = config['columns']['clinical_features'] # Keys related to the columns containing the clinical data to be used in the model

    label_manager = LabelTypesManager(config) 
    label_columns = []
    # Add the keys related to the columns that contain the labels that are to be predicted by the model
    for col in label_manager.label_names_full_list:
        if isinstance(col, (list, tuple)):
            label_columns.extend(col)
        else:
            label_columns.append(col)

    patient_ids_list = df[PATIENT_ID_COL_NAME].astype(str).tolist() # Gets the PatientID's 

    features_arr = df.set_index(PATIENT_ID_COL_NAME).loc[patient_ids_list, clinical_features_columns].values.astype(np.float32) # Gets the clinical data from the patients
    labels_arr = df.set_index(PATIENT_ID_COL_NAME).loc[patient_ids_list, label_columns].values.astype(np.float32)  # Get the labels from the patients

    data_dicts = [] # Organizes the in and output data per patient, with clinical feaures, labels, patient ids, and image paths
    for idx, patient_id in enumerate(patient_ids_list):
        patient_dict = {
            'features': features_arr[idx],
            'label_list': labels_arr[idx],
            'patient_id': patient_id
        }
        for image_key in image_keys:
            patient_dict[image_key] = os.path.join(patients_data_dir, patient_id, f"{image_key}.npy")
        data_dicts.append(patient_dict)

    return data_dicts, patient_ids_list

def make_dataloader(config : dict, df_data: pd.DataFrame, transforms, validation_mode : bool = True):
    """
    Construct PyTorch Dataset object, and then DataLoader.
    CacheDataset: caches data in RAM storage. By caching the results of deterministic / non-random preprocessing
        transforms, it accelerates the training data pipeline.
    PersistentDataset: caches data in disk storage instead of RAM storage.
    Source: https://docs.monai.io/en/stable/data.html

    Args:
        config (dict): config parameters
        df_data (pd.Dataframe): dataset to insert into this dataloader
        transforms: sequence of MONAI transforms to apply to each patient (mostly image-based)
        validation_mode (bool): 

    Returns:
        dataloader (): dataloader containing df_data's patients
        metadata (dict): dictioanry containing metadata about the input data (e.g. batch size or dimensions of the images)
    """

    dataset_type = config['data']['dataloader']['dataset_type'] 
    batch_size = config['training']['batch_size'] # if not validation_mode else 1
    num_workers = 4 # config['data']['dataloader']['num_workers'] if not validation_mode else config['data']['dataloader']['num_workers'] // 2
    persistent_workers = True if num_workers > 0 else False
    drop_last = False
    pin_memory = True if num_workers > 0 else False
    
    # check that the dataframe is not empty
    if len(df_data) == 0:
        logging.warning("The dataframe passed to make_dataloader is empty. Returning None for dataloader and metadata.")

        dataloader = None
        metadata = None
        return dataloader, metadata

    data_dict, patient_IDs_list = prepare_data_dictionaries(config, df_data)  # do some reformatting of the dataframe -> list of dictionaries
    update_dict = None

    # Define Dataset class
    if dataset_type in ['standard', None]:
        ds_class = Dataset
    elif dataset_type == 'cache':
        ds_class = CacheDataset
        update_dict = {'cache_rate': 1, 'num_workers': num_workers, "runtime_cache": 'process'}
    elif dataset_type == 'persistent':
        ds_class = PersistentDataset
        data_dir = config['paths']['images']
        cache_dir = os.path.join(os.path.dirname(data_dir), "dataloader_cache")
        #cache_dir = "/home/macraedc/cached_data"
        update_dict = {'cache_dir': cache_dir}
        os.makedirs(cache_dir, exist_ok=True)
        #create_folder_if_not_exists(cache_dir)
    elif dataset_type == 'smartcache':
        ds_class = SmartCacheDataset
        update_dict = {'cache_rate': 1 if validation_mode else config['data']['dataloader']['smartcache']['cache_rate'], 
                       'num_init_workers': None, "num_replace_workers": 
                       config['data']['dataloader']['smartcache']['num_replace_workers']}
    else:
        raise ValueError('Invalid dataset_type: {}.'.format(dataset_type))
    
    # these lines are needed to make the Dataset class work with multiple workers (prevents a memory leak)
    manager = Manager()
    data_dict = manager.list(data_dict)
    
    # Define Dataset function arguments
    ds_args_dict = {'data': data_dict, 'transform': transforms}

    # Update Dataset function arguments based on type of Dataset class
    if update_dict is not None:
        ds_args_dict.update(update_dict)

    # Initialize Dataset
    data_ds = ds_class(**ds_args_dict)
    data_ds.patient_IDs_list = patient_IDs_list
    data_ds.df = df_data
    
    # Define arguments for Dataloader class
    shuffle = False if validation_mode else True # the validation and test sets should not be shuffled, training should always be shuffled

    dl_args_dict = {'dataset': data_ds, 'batch_size': batch_size, 'shuffle': shuffle, 'sampler': None,
                    'num_workers': num_workers, 'drop_last': drop_last, 'persistent_workers': persistent_workers,
                    #'multiprocessing_context': 'spawn',
                    'pin_memory': pin_memory} #  }   # "prefetch_factor": 4,
    
    # if we want to perform mixup augmentation, we need to change the collate function 
    # (as mixup is a batch-level augmentation, it can't be done in the transforms)
    if config['data']['augmentation']['mixup']['isEnabled'] and not validation_mode:
        dl_args_dict['collate_fn'] = MixUp(config)
    
    
    # Initialize DataLoader
    dataloader = PatientDataLoader(**dl_args_dict) 

    # collect some metadata about the dimensions of the dataset
    metadata = collect_metadata(dataloader)

    return dataloader, metadata
