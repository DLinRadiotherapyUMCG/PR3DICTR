import os
import json
import math
import torch
import logging
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
    MapLabelValue,
    ToDeviced,
    CopyItemsd,
    ToTensor,
    TorchVisiond,
    MapTransform,
)
from monai.utils.type_conversion import  convert_to_tensor
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union, Mapping, Optional, Any
DtypeLike = Union[np.dtype, type, str, None]
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.enums import TransformBackends


from src.dataset.transforms.ConvertMetaTensorToTensor import ConvertMetaTensorToTensor
from src.dataset.transforms.CheckImageDimensions import CheckImageDimensions


def get_preprocessing_transforms(config: dict) -> list:
    """
    Function that returns a list of preprocessing transforms, which are used for normalising the data.
    Args:
        config: 
    Returns:
        preproc_transforms: a list of MONAI transforms
    """

    image_keys = config['data']['image_keys']
    preproc_transforms = []

    for img_key in image_keys:
        if img_key in config['data']['preprocessing']['needs_scaling']:
            preproc_transforms.append(ScaleIntensityRanged(keys=[img_key],
                                        a_min=config['data']['preprocessing'][img_key]['a_min'], a_max=config['data']['preprocessing'][img_key]['a_max'],                                         # NOTE: commented out as segmap is not used
                                        b_min=config['data']['preprocessing'][img_key]['b_min'], b_max=config['data']['preprocessing'][img_key]['b_max'],
                                        clip=config['data']['preprocessing'][img_key]['clip'])
                                    )

        elif img_key in config['data']['preprocessing']['needs_label_mapping']:
            preproc_transforms.append(MapLabelValued(
                                        keys=[img_key],
                                        orig_labels=[index for index, x in enumerate(config['data']['preprocessing'][img_key]['target_labels'])],
                                        target_labels=config['data']['preprocessing'][img_key]['target_labels']) 
                ) 
    
    return preproc_transforms




def get_random_transforms(config: dict, keys: list) -> list:
    """
    Function that returns a list of random transforms, which are used for random image augmentations.
    Args:
        config: 
        keys:
    Returns:
        random_transforms: a list of MONAI transforms
    """

    # get a list of which transformations to apply (i.e. ['random_crop', 'flip', 'affine', 'rotate', 'noise'])
    desired_augmentations = get_random_augmentation_names_from_config(config) 
    data_aug_p = config['data']['augmentation']['prob']
    #data_aug_strength = config['data']['augmentation']['strength']
    
    random_transforms = []
    for aug_name in desired_augmentations:
        
        if aug_name == 'random_crop':
            # this always gets applied
            random_transforms.append(RandSpatialCropd(keys=keys, roi_size=config['data']['preprocessing']['crop_shape'], random_size=False, random_center=True))
        
        elif aug_name == 'flip':
            random_transforms.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=-1))   # NOTE: This one is just 50% all of the time
        
        elif aug_name == 'affine':
            # TODO: NOTE: Daniel: check how these ranges are defined? I'm not sure how the new one works.
            tr_max = np.random.randint(0, config['data']['augmentation']['list']['affine']['translate_max'], 3)
            sc_max = np.random.random(3)*config['data']['augmentation']['list']['affine']['scale_max']
            sc_max[2] = sc_max[2]+config['data']['augmentation']['list']['affine']['z_scale']
            random_transforms.append(RandAffined(keys=keys, prob=data_aug_p, translate_range=tr_max, scale_range=sc_max, padding_mode='border', mode='bilinear'))
            
            # NOTE: this was the old method
            #random_transforms.append(RandAffined(keys=keys, prob=data_aug_p, translate_range=(7 * data_aug_strength,) * 3, scale_range=(0.07 * data_aug_strength,) * 3, padding_mode='border', mode='bilinear'))

        elif aug_name == 'rotate':
            rot_max = config['data']['augmentation']['list']['rotate']['max'] / 180 * 3.14  # convert degrees to radians
            random_transforms.append(RandRotated(keys=keys, prob=data_aug_p, range_x=rot_max, align_corners=True, padding_mode='border', mode='bilinear'))
        
        elif aug_name == 'noise':
            mean = config['data']['augmentation']['list']['noise']['mean']
            std = config['data']['augmentation']['list']['noise']['std']
            random_transforms.append(RandGaussianNoised(keys = keys, prob = data_aug_p, mean = mean, std = std))

        else:
            raise ValueError(f"Augmentation: {aug_name} is not implemented!")
        
    return random_transforms



def get_random_augmentation_names_from_config(config: dict) -> list:
    """
    Function that returns a list of random augmentations from the config file.
    Args:
        config (dict): 
    Returns:
        random_transforms (list): a list of strings with the names of the augmentations
    """
    
    augmentations_list_config = config['data']['augmentation']['list']

    random_transforms = []

    # loop through all of the random augmentation types in the the config
    for aug_name, aug_specific_config_dict in augmentations_list_config.items():
        
        # check if this augmentation is enabled
        if aug_specific_config_dict["isEnabled"]:
            random_transforms.append(aug_name)

    # return a list of strings with the names of the augmentations
    return random_transforms
        




def get_transforms(config: dict):
    """
    Function that makes two sets of MONAI transforms: one for training (e.g. with random augmentations) 
    and one for validation (without random augmentations).

    Args:
        config: 

    Returns:
        train_transforms: 
        val_transforms: 
    """

    image_keys = config['data']['image_keys']
    concat_key = 'input'
    

    """ Start with the generic transforms """

    # get all of the generic transforms (which apply to both the training and the validation/test sets)
    generic_transforms = Compose([
        LoadImaged(keys=image_keys, image_only=True, ensure_channel_first=False),
        CheckImageDimensions(keys=image_keys, desired_num_img_dims=config['data']['image_num_dimensions']),                                       # checks if there are enough dimensions (eg. [1,96,96,96] and not [96,96,96]), otherwise fixes it
        EnsureTyped(keys=image_keys + ['features', 'label_list', 'patient_id'], data_type='tensor'),
    ])

    # if data preprocessing is enabled, add the normalisation transform (e.g. for HNC dataset)
    if config['data']['preprocessing']['isEnabled']:
        
        preproc_transforms = get_preprocessing_transforms(config)
             
        # add the preproc_transforms to the generic transforms
        generic_transforms = Compose([
                                    generic_transforms, 
                                    Compose(preproc_transforms)
                                    ])
        
    # concatenate the images into a single tensor
    generic_transforms = Compose([
                                    generic_transforms, 
                                    ConcatItemsd(keys=image_keys, name=concat_key, dim=0),
                                    DeleteItemsd(keys=image_keys),  # removes excess memory usage
                                    ])
    
    

    """ Validation set transforms """

    # cropping the images, if preprocessing is enabled
    if config['data']['preprocessing']['isEnabled']:
        val_transforms = Compose([
            generic_transforms,
            CenterSpatialCropd(keys=[concat_key], roi_size=config['data']['preprocessing']['crop_shape'])
        ])

        # if yes preprocessing, but no (random cropping) augmentation, we need to crop the training images as well
        if (config['data']['augmentation']['isEnabled'] == False) or (config['data']['augmentation']['list']['random_crop']['isEnabled'] == False):
            train_transforms = Compose([
                                generic_transforms,
                                CenterSpatialCropd(keys=[concat_key], roi_size=config['data']['preprocessing']['crop_shape'])
                            ])
        else:
            train_transforms = generic_transforms
    # no preprocessing (or random cropping) --> both train and val have the same transforms to load the data
    else:
        train_transforms = generic_transforms
        val_transforms = generic_transforms
        
    """ 
    Random image transforms for the training set 
    """

    if config['data']['augmentation']['isEnabled']:
        
        # get a list of random MONAI transforms (for image augmentations)
        random_transforms = get_random_transforms(config, keys=concat_key)

        train_transforms = Compose([
            train_transforms,
            Compose(random_transforms)
        ])


    """ Turn the MetaTensors into normal PyTorch Tensors """
    
    train_transforms = Compose([
                                train_transforms,
                                #MONAI_MixUpd(keys=[concat_key], batch_size=config['training']['batch_size'], alpha=1.0),
                                ConvertMetaTensorToTensor(keys=[concat_key, 'label_list', 'features', 'patient_id'])
                            ])
    val_transforms = Compose([
                                val_transforms,
                                ConvertMetaTensorToTensor(keys=[concat_key, 'label_list', 'features', 'patient_id'])
                            ])

    # Flatten the transforms (removes nested Composes, turns it into one long set of transforms)
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()
    train_transforms.set_random_state(seed=config['general']['seed'])
    val_transforms.set_random_state(seed=config['general']['seed'])

    for mode, t in zip(['Train', 'Validation'], [train_transforms, val_transforms]):
        logging.debug('{} transforms:'.format(mode))
        for i in t.transforms:
            logging.debug('\t{}, keys={}'.format(i.__class__, i.keys))

    return train_transforms, val_transforms