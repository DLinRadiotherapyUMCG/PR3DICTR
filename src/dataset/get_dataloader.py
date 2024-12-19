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
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CenterSpatialCropd,
    DeleteItemsd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    Resized,
    RandSpatialCropd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGridDistortiond,
    RandRotated,
    RandZoomd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    MapLabelValued,
    ToDeviced,
    CopyItemsd,
    ToTensor,
    TorchVisiond,
)
#from monai.transforms.regularization.dictionary import MixUpd
#from utils.img_transforms import AugMix, NormaliseInputs, ConvertMetaTensorToTensor
from multiprocessing import Manager





def prepare_data_dictionaries(config, df):
    """
    Takes in a dataframe, and reformats it as a list of dictionaries
    The dictionaries contain the patient's data, including the paths to the images and the clinical features. The dictionaries are used in the Monai Datasets.
    """
    patients_data_dir = config['paths']['images']
    image_keys = config['data']['image_keys']

    clinical_features_columns = config['columns']['clinical_features']
    label_columns = config['columns']['labels']
    

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



def make_dataloader(config, df_data, transforms, validation_mode=True):
    """
    Construct PyTorch Dataset object, and then DataLoader.

    CacheDataset: caches data in RAM storage. By caching the results of deterministic / non-random preprocessing
        transforms, it accelerates the training data pipeline.
    PersistentDataset: caches data in disk storage instead of RAM storage.
    Source: https://docs.monai.io/en/stable/data.html

    Args:
        config: 
        data_dict:
        transforms:
        batch_size:
        drop_last:
        logger: logger
        print_info: whether to print information about the dataloader settings (i.e. just do it on the first time this is used, otherwise its very repetitive)

    Returns:
        dataloader
    """

    dataset_type = config['data']['dataloader']['dataset_type']   #'cache'
    dataloader_type = config['data']['dataloader']['dataloader_type']        #'standard'
    batch_size = config['training']['batch_size']
    #cache_rate = 1
    num_workers = config['data']['dataloader']['num_workers'] if not validation_mode else 1
    persistent_workers = True if num_workers > 0 else False#  config['data']['dataloader']['persistent_workers']
    drop_last = True
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
    # elif dataset_type == 'persistent':
    #     ds_class = PersistentDataset
    #     update_dict = {'cache_dir': cache_dir}
    #     create_folder_if_not_exists(cache_dir)
    elif dataset_type == 'smartcache':
        ds_class = SmartCacheDataset
        update_dict = {'cache_rate': config['data']['dataloader']['smartcache']['cache_rate'], 
                       'num_workers': num_workers, "num_replace_workers": 
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

    # if print_info:
    #     logger.my_print('Dataloader arguments:')
    #     logger.my_print('\tBatch_size: {}.'.format(batch_size))
    #     logger.my_print('\tShuffle: {}.'.format(shuffle))
    #     logger.my_print('\tSampler: {}.'.format(sampler))
    #     logger.my_print('\tNum_workers: {}.'.format(num_workers))
    #     logger.my_print('\tPersistent_workers: {}.'.format(config.persistent_workers))
    #     logger.my_print('\tPin_memory: {}.'.format(config.pin_memory))
    #     logger.my_print('\tTo_device: {}.'.format(config.to_device))
    from src.dataset.transforms.MixUp import MixUp
    dl_args_dict = {'dataset': data_ds, 'batch_size': batch_size, 'shuffle': shuffle, 'sampler': None,
                          'num_workers': num_workers, 'drop_last': drop_last, 'persistent_workers': persistent_workers,
                          'pin_memory': pin_memory,
                          
                          #'collate_fn': MixUp(config),
                          }
    
    if config['data']['augmentation']['mixup']['isEnabled'] and not validation_mode:
        dl_args_dict['collate_fn'] = MixUp(config)
    #dl_args_dict.update
    
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


from monai.data.utils import list_data_collate

def simple_collate_fn(batch):
    batch = list_data_collate(batch)

    return batch


#from src.dataset.transforms.MixUp import MONAI_MixUpd

def batch_augmentation_collate_fn(batch):
    batch = list_data_collate(batch)

    # Augment the batch
    # MIXUP HERE
    print(batch['input'].shape)


    
    return batch


