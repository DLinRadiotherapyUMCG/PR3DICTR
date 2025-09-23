import torch
import logging
import numpy as np
import pandas as pd
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CenterSpatialCropd,
    DeleteItemsd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
)
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union, Mapping, Optional, Any
DtypeLike = Union[np.dtype, type, str, None]

from src.dataset.transforms.ConvertMetaTensorToTensor import ConvertMetaTensorToTensor
from src.dataset.transforms.CheckImageDimensions import CheckImageDimensions
from src.dataset.transforms.get_preprocessing_transforms import get_preprocessing_transforms
from src.dataset.transforms.get_random_transforms import get_random_transforms

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
    generic_transforms = [
        EnsureTyped(keys = ['features', 'label_list', 'patient_id'], data_type='tensor'),  # ensure that the data is a tensor
    ]

    if image_keys:
        generic_transforms.extend([
            LoadImaged(keys=image_keys, image_only=True, ensure_channel_first=False),  # load the images
            CheckImageDimensions(keys=image_keys, desired_num_img_dims=config['data']['image_num_dimensions']),    # checks if there are enough dimensions (eg. [1,xxx,xxx,xxx] and not [xxx,xxx,xxx]), otherwise fixes it
            EnsureTyped(keys=image_keys, data_type='tensor'),
        ]
        )
    
    generic_transforms = Compose(generic_transforms)

    # if data preprocessing is enabled, add the normalisation transform (e.g. for HNC dataset)
    if config['data']['preprocessing']['isEnabled']:
        
        preproc_transforms = get_preprocessing_transforms(config)
             
        # add the preproc_transforms to the generic transforms
        generic_transforms = Compose([
                                    generic_transforms, 
                                    Compose(preproc_transforms)
                                    ])
        
    # concatenate the images into a single tensor
    if image_keys:
        generic_transforms = Compose([
                                    generic_transforms, 
                                    ConcatItemsd(keys=image_keys, name=concat_key, dim=0),
                                    DeleteItemsd(keys=image_keys),  # removes excess memory usage
                                    EnsureTyped(keys=[concat_key], data_type='tensor'), # , dtype=torch.float16),  # ensure that the data is a tensor
                                    ])
    

    """ Validation set transforms """

    # cropping the images, if preprocessing is enabled
    if config['data']['preprocessing']['isEnabled'] and image_keys:
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

    if config['data']['augmentation']['isEnabled'] and image_keys:
        
        # get a list of random MONAI transforms (for image augmentations)
        random_transforms = get_random_transforms(config, keys=concat_key)

        train_transforms = Compose([
            train_transforms,
            Compose(random_transforms)
        ])

    """ Turn the MetaTensors into normal PyTorch Tensors """
    final_transform_keys = ['label_list', 'features', 'patient_id']
    if image_keys:
        final_transform_keys.append(concat_key)
   
    train_transforms = Compose([
                                train_transforms,
                                ConvertMetaTensorToTensor(keys=final_transform_keys),
                                EnsureTyped(keys=final_transform_keys, data_type='tensor', dtype=torch.float),
                            ])
    val_transforms = Compose([
                                val_transforms,
                                ConvertMetaTensorToTensor(keys=final_transform_keys),
                                EnsureTyped(keys=final_transform_keys, data_type='tensor', dtype=torch.float),
                            ])

    # Flatten the transforms (removes nested Composes, turns it into one long set of transforms)
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()
    train_transforms.set_random_state(seed=config['general']['seed'])
    val_transforms.set_random_state(seed=config['general']['seed'])

    # Logging transform info
    for mode, t in zip(['Train', 'Validation'], [train_transforms, val_transforms]):
        logging.debug('{} transforms:'.format(mode))
        for i in t.transforms:
            logging.debug('\t{}, keys={}'.format(i.__class__, i.keys))

    return train_transforms, val_transforms