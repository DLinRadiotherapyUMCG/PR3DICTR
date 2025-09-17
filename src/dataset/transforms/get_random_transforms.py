import numpy as np
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
            random_transforms.append(RandAffined(keys=keys, prob=data_aug_p, translate_range=tr_max, scale_range=sc_max, padding_mode='zeros', mode='bilinear'))
            
            # NOTE: this was the old method
            #random_transforms.append(RandAffined(keys=keys, prob=data_aug_p, translate_range=(7 * data_aug_strength,) * 3, scale_range=(0.07 * data_aug_strength,) * 3, padding_mode='border', mode='bilinear'))

        elif aug_name == 'rotate':
            rot_max = config['data']['augmentation']['list']['rotate']['max'] / 180 * 3.14  # convert degrees to radians
            random_transforms.append(RandRotated(keys=keys, prob=data_aug_p, range_x=rot_max, align_corners=True, 
                                                 padding_mode='zeros', mode='bilinear'))
        
        elif aug_name == 'noise':
            mean = config['data']['augmentation']['list']['noise']['mean']
            std = config['data']['augmentation']['list']['noise']['std']
            random_transforms.append(RandGaussianNoised(keys = keys, prob = data_aug_p, mean = mean, std = std))

        else:
            raise ValueError(f"Augmentation: {aug_name} is not implemented!")
        
    return random_transforms