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
        # # Useful for creating figures such as attention maps
        #CopyItemsd(keys=['segmentation_map'], names=['segmentation_map_original'], times=1),
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
            # RandShiftIntensityd(keys=[concat_key], prob=data_aug_p,
            #                     offsets=(0, 0.05 * data_aug_strength), 
            #                     channel_wise=True
            #                     ),
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