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


def get_transforms(config):
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
        
        preproc_transforms = []

        for img_key in image_keys:
            if img_key in config['data']['preprocessing']['needs_scaling']:
                preproc_transforms.append(ScaleIntensityRanged(keys=['ct'],
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

        #####################################

        # if 'ct' in image_keys:
        #     preproc_transforms.append(ScaleIntensityRanged(keys=['ct'],
        #                                 a_min=config['data']['preprocessing']['ct']['a_min'], a_max=config['data']['preprocessing']['ct']['a_max'],                                         # NOTE: commented out as segmap is not used
        #                                 b_min=config['data']['preprocessing']['ct']['b_min'], b_max=config['data']['preprocessing']['ct']['b_max'],
        #                                 clip=config['data']['preprocessing']['ct']['clip'])
        #                      )
        # if 'rtdose' in image_keys:
        #     preproc_transforms.append(ScaleIntensityRanged(keys=['rtdose'],
        #                                 a_min=config['data']['preprocessing']['rtdose']['a_min'], a_max=config['data']['preprocessing']['rtdose']['a_max'],                                         # NOTE: commented out as segmap is not used
        #                                 b_min=config['data']['preprocessing']['rtdose']['b_min'], b_max=config['data']['preprocessing']['rtdose']['b_max'],
        #                                 clip=config['data']['preprocessing']['rtdose']['clip'])
        #                      )
        # if 'segmentation_map' in image_keys:
        #      preproc_transforms.append( MapLabelValued(
        #          keys=['segmentation_map'],
        #         orig_labels=[index for index, x in enumerate(config['data']['preprocessing']['segmentation_map']['target_labels'])],
        #         target_labels=config['data']['preprocessing']['segmentation_map']['target_labels']) 
        #         ) 
             
             
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
        if (config['data']['augmentation']['isEnabled'] == False) or ('crop' not in config['data']['augmentation']['augmentation_list']):
            train_transforms = Compose([
                                generic_transforms,
                                CenterSpatialCropd(keys=[concat_key], roi_size=config['data']['preprocessing']['crop_shape'])
                            ])
        else:
            train_transforms = generic_transforms

    else:
        train_transforms = generic_transforms
        val_transforms = generic_transforms
        
    """ Training set transforms """
    

    if config['data']['augmentation']['isEnabled']:
        
        desired_augmentations = config['data']['augmentation']['augmentation_list']
        data_aug_p = config['data']['augmentation']['prob']
        data_aug_strength = config['data']['augmentation']['strength']
        
        random_transforms = []
        for aug in desired_augmentations:
            
            if aug == 'crop':
                # this always gets applied
                random_transforms.append(RandSpatialCropd(keys=[concat_key], roi_size=config['data']['preprocessing']['crop_shape'], random_size=False, random_center=True))
            
            elif aug == 'flip':
                random_transforms.append(RandFlipd(keys=[concat_key], prob=data_aug_p, spatial_axis=1))   # NOTE: should this just be 50% all of the time?
            
            elif aug == 'affine':
                random_transforms.append(RandAffined(keys=[concat_key], prob=data_aug_p, translate_range=(7 * data_aug_strength,) * 3, scale_range=(0.07 * data_aug_strength,) * 3, padding_mode='border', mode='bilinear'))
            
            elif aug == 'rotate':
                random_transforms.append(RandRotated(keys=[concat_key], prob=data_aug_p, range_x=(3.14 / 24 * data_aug_strength), align_corners=True, padding_mode='border', mode='bilinear'))
            
            else:
                raise ValueError("Augmentation: " + aug + " is not implemented!")

        train_transforms = Compose([
            train_transforms,
            Compose(random_transforms)
        ])


    """ Turn the MetaTensors into normal PyTorch Tensors """
    train_transforms = Compose([
                                train_transforms,
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
    


