"""
Load data
"""
import os
import json
import math
import torch
import random
import logging
import numpy as np
import pandas as pd
from monai.data import Dataset, CacheDataset, PersistentDataset, GDSDataset, DataLoader, ThreadDataLoader, ThreadBuffer, SmartCacheDataset
from multiprocessing import Manager


from src.dataset.LabelTypesManager import LabelTypesManager
from src.dataset.transforms.MixUp import MixUp



def prepare_data_dictionaries(config: dict, df: pd.DataFrame):
    """
    Hehlper function that takes in a dataframe, and reformats it as a list of dictionaries
    The dictionaries contain the patient's data, including the paths to the images and the clinical features. The dictionaries are used in the Dataset classes.
    Args:
        config (dict): configuration object
        df (pd.DataFrame): dataframe for this dataloader's (train, val or test) dataset
    Returns:
        data_dicts (list of dicts): a list of dictionaries, where each dictionary is one patient
        patient_ids_list (list): list of patientIDs in this dataset
    """
    patients_data_dir = config['paths']['images']
    image_keys = config['data']['image_keys']
    clinical_features_columns = config['columns']['clinical_features']
    label_columns = config['columns']['labels']

    LabelManager = LabelTypesManager(config)  # create a LabelTypesManager object to handle the labels
    label_columns = LabelManager.label_names_full_list  # get the full list of labels, (including event and days labels for event endpoints)
    #label_columns, _ = check_label_types(config, label_columns)  # check if the label types in the config match the endpoint list
    print("LABELS:", label_columns)
    # Flatten label_columns in case it contains tuples
    flattened_label_columns = []
    for col in label_columns:
        if isinstance(col, (list, tuple)):
            flattened_label_columns.extend(col)
        else:
            flattened_label_columns.append(col)
    label_columns = flattened_label_columns
    
    
    patient_ids_list = [str(patient_id) for patient_id in df['PatientID']]

    features_list = [np.array([df[df['PatientID'] == patient_id][feature].values[0] for feature in
                             clinical_features_columns]).astype(np.float32) \
                            for patient_id in patient_ids_list]

    label_values_list = [np.array(df[df['PatientID'] == patient_id][label_columns].values[0]).astype(np.float32) \
                            for patient_id in patient_ids_list]
    
    # make a dictionary for each patient's data
    data_dicts = [
            {#'ct': os.path.join(patients_data_dir, patient_id, config['data']['filenames']['ct']),
            #'rtdose': os.path.join(patients_data_dir, patient_id, config['data']['filenames']['rtdose']),
            #'segmentation_map': os.path.join(patients_data_dir, patient_id, config['data']['filenames']['segmentation_map']),
            'features': feature_values,
            'label_list': label_values,
            'patient_id': patient_id}
            for patient_id, feature_values, label_values in zip(patient_ids_list, features_list, label_values_list)
        ]
    
    # add each of the images we want to include (CT, RTDOSE, Segmentation map)
    # this method makes it possible to not include one of them if we don't want to (I'm looking at you, segmentation map)
    for idx, patient_id in enumerate(patient_ids_list):
        for image_key in image_keys:
            data_dicts[idx][image_key] = os.path.join(patients_data_dir, patient_id, image_key + ".npy")
    
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

    dataset_type = config['data']['dataloader']['dataset_type'] # if not validation_mode else 'cache'
    dataloader_type = config['data']['dataloader']['dataloader_type']        #'standard'
    batch_size = config['training']['batch_size'] if not validation_mode else 1
    #cache_rate = 1
    num_workers = config['data']['dataloader']['num_workers'] if not validation_mode else config['data']['dataloader']['num_workers'] // 2
    persistent_workers = True if num_workers > 0 else False#  config['data']['dataloader']['persistent_workers']
    drop_last = False
    pin_memory = True if num_workers > 0 else False
    
    
    dataset_size = len(df_data)
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

    # Define DataLoader class
    from src.dataset.ToxDataLoader import ToxDataLoader
    if dataloader_type in ['standard', None]:
        dl_class = DataLoader
    elif dataloader_type == 'thread':
        dl_class = ThreadDataLoader
    else:
        raise ValueError('Invalid dataloader_type: {}.'.format(dataloader_type))
    
    dl_class = ToxDataLoader

    # Define Dataloader function arguments
    if validation_mode:   # the validation and test datasets should not be shuffled
        shuffle = False
    else:                 # the training dataset should always be shuffled
        shuffle = True

    
    dl_args_dict = {'dataset': data_ds, 'batch_size': batch_size, 'shuffle': shuffle, 'sampler': None,
                          'num_workers': num_workers, 'drop_last': drop_last, 'persistent_workers': persistent_workers,
                          'pin_memory': pin_memory, "prefetch_factor": 2} #  }   # "prefetch_factor": 4,
    
    # if we want to perform mixup augmentation, we need to change the collate function 
    # (as mixup is a batch-level augmentation, it can't be done in the transforms)
    if config['data']['augmentation']['mixup']['isEnabled'] and not validation_mode:
        dl_args_dict['collate_fn'] = MixUp(config)
    
    
    # Initialize DataLoader
    if dataset_size > 0:
        dataloader = dl_class(**dl_args_dict) 

        example_data = next(iter(dataloader))
        
        batch_size1, channels, depth, height, width = example_data['input'].shape
        batch_size2, n_features = example_data['features'].shape

        #print("features", example_data['features'].shape)
        #print("label_list", example_data['label_list'].shape)
        batch_size3, n_labels = example_data['label_list'].shape
        
        assert batch_size1 == batch_size2 == batch_size3

        metadata = {
            "channels": channels,
            "depth": depth,
            "height": height,
            "width": width,
            "n_features": n_features,
            "n_labels": n_labels,
        }
    else:
        dataloader, metadata = None, None
    
    return dataloader, metadata




