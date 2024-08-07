"""
Load data
"""
import os
import json
import math
import torch
import random
import numpy as np
import pandas as pd
from monai.data import Dataset, CacheDataset, PersistentDataset, GDSDataset, DataLoader, ThreadDataLoader, ThreadBuffer
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
from utils.img_transforms import AugMix, NormaliseInputs, ConvertMetaTensorToTensor
from multiprocessing import Manager

#import config
import misc
import data_preproc.data_preproc_config as data_preproc_config
from misc import get_all_df_stats#, stratified_sampling_split
from data_preproc.data_preproc_functions import set_default, create_folder_if_not_exists
from stratified_sampling import stratified_sampling_split


def perform_validation_stratified_sampling(config, df, frac, output_path_filename, logger):
    """
    Create stratified samples for validation and test set.

    Important, apply this function inside get_files() (because get_files() will be run earlier if we perform CV)

    Source: https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/groupby/groupby.py#L3687-L3813
    Source: sample.sample(): https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/sample.py
    Source: random_state.choice(): https://github.com/numpy/numpy/blob/main/numpy/random/_generator.pyx#L601-L862

    Args:
        df:
        frac:
        strata_groups:
        split_col:
        output_path_filename:
        seed:
        logger:

    Returns: dataframe with column 'Split' indicating whether the patient_id belongs to 'train', 'val' or 'test'

    """
    split_col = config.split_col

    # Train - Val split
    df_train_val = df[df[split_col] == 'train_val'].copy()
    df_test = df[df[split_col] == 'test'].copy()
    df_train, df_val = stratified_sampling_split(df=df_train_val, frac=frac, logger=logger, name='validation')

    assert len(df) == len(df_train.index.tolist() + df_val.index.tolist() + df_test.index.tolist())
    assert len(df) == len(set(df_train.index.tolist() + df_val.index.tolist() + df_test.index.tolist()))

    # Create column indicating whether the patient_id belongs to 'train', 'val' or 'test'
    df_train[split_col] = 'train'
    df_val[split_col] = 'val'
    df_test[split_col] = 'test'

    # Save df_split
    df_split = pd.concat([df_train, df_val, df_test]).sort_index()
    df_split.to_csv(output_path_filename, sep=';', index=False)

def get_test_set_files(config, logger):
    # Initialize variables
    patient_id_col = data_preproc_config.patient_id_col
    patient_id_length = data_preproc_config.patient_id_length
    patients_data_dir = config.patients_data_dir
    split_col = config.split_col

    # Load features
    df_features = pd.read_csv(os.path.join(config.data_dir, config.filename_stratified_sampling_test_csv), sep=';', decimal='.')
    df_features[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_features[patient_id_col]]

    # Create list of images and labels_raw_train
    label_values_list, features_list, patient_ids_list =  list(), list(), list()

    # Get PatientIDs
    patient_ids_list = df_features[patient_id_col].values.tolist()
    # sort() to make sure that different platforms use the same order --> required for random.shuffle() later
    patient_ids_list.sort()
    for patient_id in patient_ids_list:
        # Patient's features data
        df_features_i = df_features[df_features[patient_id_col] == patient_id]
        # Features
        features_list += [[float(str(x).replace(',', '.')) for x in y] for y in
                          df_features_i[config.features_dl].values.tolist()]
        # Endpoint
        label_values_list.append([int(df_features_i[x].iloc[0]) for x in config.endpoint_list])

    assert len(patient_ids_list) == len(features_list) == len(label_values_list)
    # Note: '0' in front of string is okay: int('0123') will become 123
    filename_segmap = data_preproc_config.filename_multihot_encoded_segmentation_map_npy if config.use_multihot_segmap else data_preproc_config.filename_segmentation_map_npy
    data_dicts = [
        {'ct': os.path.join(patients_data_dir, patient_id, data_preproc_config.filename_ct_npy),
         'rtdose': os.path.join(patients_data_dir, patient_id, data_preproc_config.filename_rtdose_npy),
         'segmentation_map': os.path.join(patients_data_dir, patient_id, filename_segmap),
         'features': feature_values,
         'label_list': label_values,
         'patient_id': patient_id}
        for patient_id, feature_values, label_values in zip(patient_ids_list, features_list, label_values_list)
    ]

    # Training-internal_validation-test data split
    patients_test = df_features[df_features[split_col] == 'test'][patient_id_col].tolist()
    
    test_dict = [x for x in data_dicts if x['patient_id'] in patients_test]

    return test_dict



def get_files(config, logger):
    """
    Fetch data files and return the training, internal validation and test files. The data directory should be
    organized as follows:
    - data_dir:
        stratified_sampling_test.csv  # with column 'Split' containing values 'train_val' (cross-validation) and 'test'
        - patients_data_dir:
            - patient_0
                - foo.npy
                - foofoo1.npy
                    ...
                - foofoofoofooN.npy
            - patient_1
                - bar.npy
                - barbar1.npy
                    ...
                - barbarbarbarN.npy

    Args:
        config:
        logger:

    Returns:
        Training, internal validation and test files.
    """
    # Initialize variables
    patient_id_col = data_preproc_config.patient_id_col
    patient_id_length = data_preproc_config.patient_id_length
    patients_data_dir = config.patients_data_dir
    split_col = config.split_col

    # Load features
    df_features = pd.read_csv(os.path.join(config.data_dir, config.filename_stratified_sampling_test_csv), sep=';', decimal='.')
    df_features[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_features[patient_id_col]]

    # Create list of images and labels_raw_train
    label_values_list, features_list, patient_ids_list = list(), list(), list()

    # Get PatientIDs
    patient_ids_list = df_features[patient_id_col].values.tolist()
    # sort() to make sure that different platforms use the same order --> required for random.shuffle() later
    patient_ids_list.sort()
    for patient_id in patient_ids_list:
        # Patient's features data
        df_features_i = df_features[df_features[patient_id_col] == patient_id]
        # Features
        features_list += [[float(str(x).replace(',', '.')) for x in y] for y in
                          df_features_i[config.features_dl].values.tolist()]
        # Endpoint
        label_values_list.append([int(df_features_i[x].iloc[0]) for x in config.endpoint_list])

    assert len(patient_ids_list) == len(features_list) == len(label_values_list)
    # Note: '0' in front of string is okay: int('0123') will become 123
    filename_segmap = data_preproc_config.filename_multihot_encoded_segmentation_map_npy if config.use_multihot_segmap else data_preproc_config.filename_segmentation_map_npy
    data_dicts = [
        {'ct': os.path.join(patients_data_dir, patient_id, data_preproc_config.filename_ct_npy),
         'rtdose': os.path.join(patients_data_dir, patient_id, data_preproc_config.filename_rtdose_npy),
         'segmentation_map': os.path.join(patients_data_dir, patient_id, filename_segmap),
         'features': feature_values,
         'label_list': label_values,
         'patient_id': patient_id}
        for patient_id, feature_values, label_values in zip(patient_ids_list, features_list, label_values_list)
    ]

    # Whether to perform random split or to perform stratified sampling
    if config.sampling_type == 'random':
        # Select random subset for testing
        if config.perform_test_run:
            data_dicts = data_dicts[:config.n_samples]

        # Training-internal_validation-test data split
        patients_test = df_features[df_features[split_col] == 'test'][patient_id_col].tolist()
        train_val_dict = [x for x in data_dicts if x['patient_id'] not in patients_test]
        n_train = round(len(train_val_dict) * config.train_frac)

        train_dict = train_val_dict[:n_train]
        val_dict = train_val_dict[n_train:]
        test_dict = [x for x in data_dicts if x['patient_id'] in patients_test]

    elif config.sampling_type == 'stratified':
        # Stratified sampling
        path_filename = os.path.join(config.data_dir, config.filename_stratified_sampling_full_csv)
        if not os.path.isfile(path_filename) or config.perform_stratified_sampling_full:
            # Create stratified_sampling.csv if file does not exists, or recreate if requested
            logger.my_print('Creating {}.'.format(path_filename))
            # Create stratified samples for train and validation set
            perform_validation_stratified_sampling(config=config, df=df_features, frac=config.val_frac,
                                        output_path_filename=path_filename,  logger=logger)

        # Load list of patients with split info
        df_split = pd.read_csv(path_filename, sep=';')
        df_split[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_split[patient_id_col]]

        # Print stats
        get_all_df_stats(config=config, df_split=df_split, logger=logger)

        train_dict, val_dict, test_dict = list(), list(), list()
        for d_dict in data_dicts:
            patient_id = d_dict['patient_id']
            df_split_i = df_split[df_split[patient_id_col] == patient_id]
            assert len(df_split_i) == 1
            split_i = df_split_i['Split'].values[0]
            if split_i == 'train':
                train_dict.append(d_dict)
            elif split_i == 'val':
                val_dict.append(d_dict)
            elif split_i == 'test':
                test_dict.append(d_dict)
            else:
                raise ValueError('Invalid split_i: {}.'.format(split_i))

        assert len(df_features) == len(data_dicts) == len(df_split) == len(train_dict) + len(val_dict) + len(test_dict)

        # Select random subset for testing
        if config.perform_test_run:
            n_samples = config.n_samples
            n_train = round(n_samples * config.train_frac)
            n_val = math.ceil(n_samples * config.val_frac)
            n_test = n_samples - n_train - n_val

            random.shuffle(train_dict)
            random.shuffle(val_dict)
            random.shuffle(test_dict)

            train_dict = train_dict[:n_train]
            val_dict = val_dict[:n_val]
            test_dict = test_dict[:n_test]
        

    else:
        raise ValueError('Invalid sampling_type: {}.'.format(config.sampling_type))

    return train_dict, val_dict, test_dict


def get_files_stats(config, train_dict, val_dict, test_dict, logger):
    nr_of_decimals = config.nr_of_decimals
    endpoint_list = config.endpoint_list
    features = config.features_dl

    logger.my_print('Extra features for Deep Learning model: {}.'.format(features))

    n_train = len(train_dict)
    n_val = len(val_dict)
    if test_dict is not None:
        n_test = len(test_dict)
    else:
        n_test = 0
    n_features = len(features)

    logger.my_print('Total number of data samples: {}.'.format(n_train + n_val + n_test))

    # Print endpoint distribution
    n_train_0_list, n_train_1_list, n_val_0_list, n_val_1_list, n_test_0_list, n_test_1_list= (list(), list(), list(), list(), list(), list())

    for i, endpoint in enumerate(endpoint_list):
        n_train_0_list.append(sum([True for x in train_dict if x['label_list'][i] == 0]))
        n_train_1_list.append(sum([True for x in train_dict if x['label_list'][i] == 1]))
        n_val_0_list.append(sum([True for x in val_dict if x['label_list'][i] == 0]))
        n_val_1_list.append(sum([True for x in val_dict if x['label_list'][i] == 1]))
        if test_dict is not None:
            n_test_0_list.append(sum([True for x in test_dict if x['label_list'][i] == 0]))
            n_test_1_list.append(sum([True for x in test_dict if x['label_list'][i] == 1]))

        logger.my_print('Endpoint: {}'.format(endpoint))
        if n_train > 0:
            logger.my_print('\tTraining size (labels=0): {}/{} ({}).'.format(n_train_0_list[-1], n_train,
                                                                           round(n_train_0_list[-1] / n_train,
                                                                                 nr_of_decimals)))
            logger.my_print('\tTraining size (labels=1): {}/{} ({}).'.format(n_train_1_list[-1], n_train,
                                                                           round(n_train_1_list[-1] / n_train,
                                                                                 nr_of_decimals)))
        if n_val > 0:
            logger.my_print('\tInternal validation size (labels=0): {}/{} ({}).'.format(n_val_0_list[-1], n_val,
                                                                                      round(n_val_0_list[-1] / n_val,
                                                                                            nr_of_decimals)))
            logger.my_print('\tInternal validation size (labels=1): {}/{} ({}).'.format(n_val_1_list[-1], n_val,
                                                                                      round(n_val_1_list[-1] / n_val,
                                                                                            nr_of_decimals)))
        if n_test > 0:
            logger.my_print('\tTest size (labels=0): {}/{} ({}).'.format(n_test_0_list[-1], n_test,
                                                                       round(n_test_0_list[-1] / n_test,
                                                                             nr_of_decimals)))
            logger.my_print('\tTest size (labels=1): {}/{} ({}).'.format(n_test_1_list[-1], n_test,
                                                                       round(n_test_1_list[-1] / n_test,
                                                                             nr_of_decimals)))
        logger.my_print('')

    return n_train_0_list, n_train_1_list, n_val_0_list, n_val_1_list, n_test_0_list, n_test_1_list, n_features


def save_patient_ids(config, train_dict, val_dict, test_dict, filename):
    """
    Save patient_ids in train_dict, val_dict and test_dict to json file. This information can be used
    for logistic regression.

    Args:
        config:
        train_dict:
        val_dict:
        test_dict:
        filename:

    Returns:

    """
    # Initialize variables
    patient_ids_dict = dict()
    files = [train_dict, val_dict]
    tags = ['train', 'val']

    if test_dict is not None:
        files.append(test_dict)
        tags.append('test')

    # Create dictionary of patient_ids
    for file, tag in zip(files, tags):
        patient_id = [x['patient_id'] for x in file]
        patient_id.sort()
        patient_ids_dict[tag] = patient_id

    # Make sure that there are no overlapping of patient_ids between train, internal_validation and test set
    assert len(misc.intersection(patient_ids_dict['train'], patient_ids_dict['val'])) == 0
    if test_dict is not None:
        assert len(misc.intersection(patient_ids_dict['train'], patient_ids_dict['test'])) == 0
        assert len(misc.intersection(patient_ids_dict['val'], patient_ids_dict['test'])) == 0

    # Save patient_ids_dict
    with open(os.path.join(config.exp_dir, filename), 'w') as f:
        json.dump(patient_ids_dict, f, default=set_default)



def get_transforms(config, logger):
    """
    Transforms for training, internal validation and test data.

    Source: https://github.com/Project-MONAI/tutorials/blob/master/modules/3d_image_transforms.ipynb
    Source: https://docs.monai.io/en/latest/transforms.html?highlight=Rand3DElastic#monai.transforms.Rand3DElastic

    Note: MONAI documentation describes that the transformers require 3D data to have shape (nchannels, H, W, D), but
        shapes (1, H, W, D) and (H, W, D) results in same augmented data, i.e. shape (H, W, D) is fine.

    LoadImaged: https://docs.monai.io/en/latest/transforms.html?highlight=loadimaged#monai.transforms.LoadImage
    EnsureTyped: https://docs.monai.io/en/latest/transforms.html?highlight=ensuretyped#monai.transforms.EnsureType
    Resized: https://docs.monai.io/en/latest/transforms.html?highlight=resized#monai.transforms.Resize
    ToDeviced: Move PyTorch Tensor to the specified device. It can help cache data into GPU and execute on GPU directly.
        CacheDataset caches the transform results until ToDeviced, so it is in GPU memory. Then in every epoch, the
        program fetches cached data from GPU memory and only execute the transforms after `ToDeviced` on GPU directly.
    Source: https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md
    Source: https://docs.monai.io/en/latest/transforms.html?highlight=todeviced#monai.transforms.ToDevice

    Args:
        config:
        logger:

    Returns:

    """

    perform_data_aug = config.perform_data_aug
    data_aug_strength = config.data_aug_strength
    data_aug_p = config.data_aug_p
    
    rand_cropping_size = config.rand_cropping_size
    input_size = config.input_size
    resize_mode = config.resize_mode
    seg_orig_labels = config.seg_orig_labels
    seg_target_labels = config.seg_target_labels

    device = config.device
    to_device = config.to_device
    image_keys = config.image_keys
    concat_key = config.concat_key
    # Define variables for exceptions

    modes_3d = [config.ct_dose_seg_interpol_mode_3d]
    modes_2d = [config.ct_dose_seg_interpol_mode_2d]
    align_corners_exception_3d = [True if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None for mode in
                                  modes_3d]
    align_corners_exception_2d = [True if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None for mode in
                                  modes_2d]
    # CT
    range_value_ct = config.ct_a_max - config.ct_a_min
    # RTDOSE
    range_value_rtdose = config.rtdose_a_max - config.rtdose_a_min
    # Segmentation_map
    # values_filter = [data_preproc_config.structures_values[x] for x in config.segmentation_structures]

    # Define generic transforms
    generic_transforms = Compose([
        LoadImaged(keys=image_keys, image_only=True),
        EnsureTyped(keys=image_keys + ['features', 'label_list'], data_type='tensor'),
        # Clip
        ScaleIntensityRanged(keys=['ct'],
                             a_min=config.ct_a_min, a_max=config.ct_a_max,
                             b_min=config.ct_a_min, b_max=config.ct_a_max,
                             clip=config.ct_clip),
        ScaleIntensityRanged(keys=['rtdose'],
                             a_min=config.rtdose_a_min, a_max=config.rtdose_a_max,
                             b_min=config.rtdose_a_min, b_max=config.rtdose_a_max,
                             clip=config.rtdose_clip),
        # Useful for creating figures such as attention maps
        # CopyItemsd(keys=['segmentation_map'], names=['segmentation_map_original'], times=1),
        MapLabelValued(keys=['segmentation_map'], orig_labels=seg_orig_labels, target_labels=seg_target_labels),
        ConcatItemsd(keys=image_keys, name=concat_key, dim=0),
        DeleteItemsd(keys=image_keys),
    ])

    # Define training transforms
    train_transforms = generic_transforms

    # Define internal validation and test transforms
    val_transforms = generic_transforms

    # Data augmentation
    if perform_data_aug:
        train_transforms = Compose([
            train_transforms,
            RandSpatialCropd(keys=[concat_key], roi_size=rand_cropping_size, random_center=True, random_size=False),
            RandFlipd(keys=[concat_key], prob=0.5, spatial_axis=-1),  # 3D: (num_channels, H[, W, …, ])
            RandAffined(keys=[concat_key], prob=data_aug_p,
                        translate_range=(7 * data_aug_strength, 7 * data_aug_strength, 7 * data_aug_strength),
                        padding_mode='zeros', mode=modes_2d),  # 3D: (num_channels, H, W[, D])
            RandAffined(keys=[concat_key], prob=data_aug_p,
                        scale_range=(0.07 * data_aug_strength, 0.07 * data_aug_strength, 0.07 * data_aug_strength),
                        padding_mode='zeros', mode=modes_2d),  # 3D: (num_channels, H, W[, D])
            # RandZoomd(keys=concat_key, prob=data_aug_p, min_zoom=[1] + [1 - 0.07 * data_aug_strength] * 2,
            #           max_zoom=[1] + [1 + 0.07 * data_aug_strength] * 2, align_corners=align_corners_exception_3d,
            #           padding_mode='edge', mode=modes_3d),  # Is similar to  RandAffined with scale_range param.
            # 3D: (nchannels, H, W, D). Only `trilinear` possible for 5D input data
            RandRotated(keys=[concat_key], prob=data_aug_p, range_x=(np.pi / 24) * data_aug_strength,
                        align_corners=align_corners_exception_2d, padding_mode='border', mode=modes_2d),
            # 3D: (nchannels, H, W, D). Only `bicubic`, `nearest` and `bilinear` possible. `Trilinear` not available because
            # the rotation is done in 2 dimensions, so that every slice is rotated by the same amount.
            # Rand3DElasticd(keys=[concat_key], prob=data_aug_p, sigma_range=(5, 8), magnitude_range=(0, round(1 * data_aug_strength)),
            #                padding_mode='border', mode=modes_3d),  # 3D: (nchannels, H, W, D)
            RandShiftIntensityd(keys=[concat_key], prob=data_aug_p,
                                offsets=(0, 0.05 * data_aug_strength), 
                                channel_wise=True
                                ),
            # RandGaussianNoised(keys=['ct'], prob=data_aug_p, mean=0.0, std=(0.1 / (range_value_ct ** (1 / 2))) * data_aug_strength),
            # RandGaussianNoised(keys=['rtdose'], prob=data_aug_p, mean=0.0, std=(0.1 / (range_value_rtdose ** (1 / 2))) * data_aug_strength),
        ])

    # Resize images
    if list(rand_cropping_size) != list(input_size) or not perform_data_aug:
        train_transforms = Compose([
            train_transforms,
            CenterSpatialCropd(keys=[concat_key], roi_size=input_size),
        ])

    val_transforms = Compose([
        val_transforms,
        CenterSpatialCropd(keys=[concat_key], roi_size=input_size),
    ])

    # DANIEL
    if config.perform_augmix:
        train_transforms = Compose([
            train_transforms,
            AugMix(keys=[concat_key], config=config)
        ])

    
    #Finally, a normalisation transform step (to normalise the model's input images to [0,1])
    train_transforms = Compose([
        train_transforms,
        NormaliseInputs(keys=[concat_key], config=config),
        ConvertMetaTensorToTensor(keys=[concat_key, 'label_list', 'features'])
    ])
    val_transforms = Compose([
        val_transforms,
        NormaliseInputs(keys=[concat_key], config=config),
        ConvertMetaTensorToTensor(keys=[concat_key, 'label_list', 'features'])
    ])

    # To device
    if to_device:
        train_transforms = Compose([
            train_transforms,
            ToDeviced(keys=[concat_key], device=device),
        ])

        val_transforms = Compose([
            val_transforms,
            ToDeviced(keys=[concat_key], device=device),
        ])

    # Flatten transforms
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    # Print transforms
    for mode, t in zip(['Train', 'Validation'], [train_transforms, val_transforms]):
        logger.my_print('{} transforms:'.format(mode))
        for i in t.transforms:
            logger.my_print('\t{}, keys={}'.format(i.__class__, i.keys))

    # set random seeds
    #misc.set_random_seeds(config, set_new_random_seed=False)
    train_transforms.set_random_state(seed=config.seed)
    val_transforms.set_random_state(seed=config.seed)

    return train_transforms, val_transforms


def get_dataloader(config, data_dict, transforms, batch_size, drop_last, logger, print_info=False, disable_shuffle=False):
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

    dataset_type = config.dataset_type
    cache_rate = config.cache_rate
    num_workers = config.num_workers
    cache_dir = config.cache_dir
    dataloader_type = config.dataloader_type
    use_sampler = config.use_sampler
    
    dataset_size = len(data_dict)
    update_dict = None

    if print_info:
        logger.my_print('Dataset type: {}.'.format(dataset_type))
        logger.my_print('Dataloader type: {}.'.format(dataloader_type))

    # Define Dataset class
    if dataset_type in ['standard', None]:
        ds_class = Dataset
    elif dataset_type == 'cache':
        ds_class = CacheDataset
        update_dict = {'cache_rate': cache_rate, 'num_workers': num_workers, "runtime_cache": config.runtime_cache}
    elif dataset_type == 'persistent':
        ds_class = PersistentDataset
        update_dict = {'cache_dir': cache_dir}
        create_folder_if_not_exists(cache_dir)
    elif dataset_type == 'GDS':
        ds_class = GDSDataset
        update_dict = {'cache_dir': cache_dir, 'device': config.device}
        create_folder_if_not_exists(cache_dir)
    else:
        raise ValueError('Invalid dataset_type: {}.'.format(dataset_type))
    
    
    manager = Manager()
    data_dict = manager.list(data_dict)

    # Define Dataset function arguments
    ds_args_dict = {'data': data_dict, 'transform': transforms}

    # Update Dataset function arguments based on type of Dataset class
    if update_dict is not None:
        ds_args_dict.update(update_dict)

    # if doing hyperparameter tuning, set progress=False to supress the progress bars in the slurm/log file
    if config.hyperparameter_tuning_mode and dataset_type == 'cache':
        ds_args_dict['progress'] = False

    # Initialize Dataset
    data_ds = ds_class(**ds_args_dict)

    # Define DataLoader class
    if dataloader_type in ['standard', None]:
        dl_class = DataLoader
    elif dataloader_type == 'thread':
        dl_class = ThreadDataLoader
    else:
        raise ValueError('Invalid dataloader_type: {}.'.format(dataloader_type))

    # Define Dataloader function arguments
    # Shuffle is not necessary for val_dl and test_dl, but shuffle can be useful for plotting random patients in main.py
    # Weighted random sampler
    if use_sampler:
        # Source: https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
        if print_info:
            logger.my_print('Using WeightedRandomSampler.')
        shuffle = False

        labels_raw_train = np.array([x['label_list'] for x in data_dict])
        weights = 1 / np.array([np.count_nonzero(1 - labels_raw_train), np.count_nonzero(labels_raw_train)])
        samples_weight = np.array([weights[t] for t in labels_raw_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        shuffle = True
        sampler = None

    if disable_shuffle: 
        shuffle = False

    if print_info:
        logger.my_print('Dataloader arguments:')
        logger.my_print('\tBatch_size: {}.'.format(batch_size))
        logger.my_print('\tShuffle: {}.'.format(shuffle))
        logger.my_print('\tSampler: {}.'.format(sampler))
        logger.my_print('\tNum_workers: {}.'.format(num_workers))
        logger.my_print('\tPersistent_workers: {}.'.format(config.persistent_workers))
        logger.my_print('\tPin_memory: {}.'.format(config.pin_memory))
        logger.my_print('\tTo_device: {}.'.format(config.to_device))
    dl_args_dict = {'dataset': data_ds, 'batch_size': batch_size, 'shuffle': shuffle, 'sampler': sampler,
                          'num_workers': num_workers, 'drop_last': drop_last, 'persistent_workers': config.persistent_workers,
                          'pin_memory': config.pin_memory}
    
    # Initialize DataLoader
    dataloader = dl_class(**dl_args_dict) if dataset_size > 0 else None
    
    return dataloader



def get_test_set_dataloader(config, fold, logger):
    # load the dataframe here
    train_dict, val_dict = {}, {} # empty dicts to make the 'save_patient_ids' function wor
    test_dict = get_test_set_files(config, logger)

    # Save list of patient_ids in training, internal validation and test
    save_patient_ids(config, train_dict, val_dict, test_dict,
                     filename=config.filename_train_val_test_patient_ids_json.format(fold))
    
    # Define training, internal validation and test transforms
    _, val_transforms = get_transforms(config=config, logger=logger)

    # make the dataloader here
    test_dl = get_dataloader(config, test_dict, val_transforms, config.batch_size, drop_last=False, logger=logger, print_info=True, disable_shuffle=True)

    return test_dl
    


def main(config, train_dict, val_dict, test_dict, fold, drop_last, logger, save_patient_ids_file=True):
    """
    Loads datasets and returns the batches corresponding to the given parameters.

    Args:
        config:
        train_dict:
        val_dict:
        test_dict:
        fold:
        drop_last:
        logger:

    Returns:

    """
    batch_size = config.batch_size
    #misc.set_random_seeds(config, set_new_random_seed=False)

    # Make sure that they are no None's, or all None's (i.e. whether we perform cross-validation or not)
    nr_nones = (train_dict is None) + (val_dict is None) + (test_dict is None)
    assert nr_nones in [0, 1, 3]

    # Get training, internal validation and test files
    if nr_nones == 3:
        train_dict, val_dict, test_dict = get_files(logger=logger)

    # for subsampling the training set (to analyse effect of training set size on model performance)
    if config.subsample_train_set:
        random.shuffle(train_dict)
        num_train_samples = len(train_dict)
        subsample_num_train_samples = round(config.subsample_train_set_size * num_train_samples)
        train_dict = train_dict[:subsample_num_train_samples]

    train_0_list, train_1_list, val_0_list, val_1_list, test_0_list, test_1_list, n_features = (
        get_files_stats(config, train_dict, val_dict, test_dict, logger))

    # Save list of patient_ids in training, internal validation and test
    if save_patient_ids_file:
        save_patient_ids(config, train_dict, val_dict, test_dict,
                        filename=config.filename_train_val_test_patient_ids_json.format(fold))

    # Define training, internal validation and test transforms
    train_transforms, val_transforms = get_transforms(config=config, logger=logger)
    
    train_dl = get_dataloader(config, train_dict, train_transforms, batch_size, drop_last, logger, print_info=True)
    val_dl = get_dataloader(config, val_dict, val_transforms, 1, drop_last, logger, print_info=False, disable_shuffle=True)
    if test_dict is not None:
        test_dl = get_dataloader(config, test_dict, val_transforms, 1, drop_last, logger, print_info=False, disable_shuffle=True)
    else:
        test_dl = None

    # Get input shape      # BUG: sometimes getting an example_data from the dataloader causes the code to hang here
    # if sum(train_0_list) + sum(train_1_list) > 0:
    #     example_data = next(iter(train_dl))
    # elif sum(val_0_list) + sum(val_1_list) > 0:
    #     example_data = next(iter(val_dl))
    # elif sum(test_0_list) + sum(test_1_list) > 0:
    #     example_data = next(iter(test_dl))
        
    # example_input = example_data['ct_dose_seg']
    #print(example_input.shape)
    #batch_size, channels, depth, height, width = example_input.shape
    #batch_size = config.batch_size
    #channels = len(config.image_keys)

    channels = len(config.image_keys)
    [depth, height, width] = config.input_size 



    return (train_dl, val_dl, test_dl, batch_size, channels, depth, height, width, 
            n_features, train_dict, val_dict, test_dict)