from monai.transforms import (
    ScaleIntensityRanged,
    MapLabelValued,
)

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

    # Goes through the different images and sets the required transform per volume types based on the input
    for img_key in image_keys:
        if img_key in config['data']['preprocessing']['needs_scaling']: # Any windowing and rescaling for the input
            preproc_transforms.append(ScaleIntensityRanged(keys=[img_key],
                                        a_min=config['data']['preprocessing'][img_key]['a_min'], a_max=config['data']['preprocessing'][img_key]['a_max'],                                         # NOTE: commented out as segmap is not used
                                        b_min=config['data']['preprocessing'][img_key]['b_min'], b_max=config['data']['preprocessing'][img_key]['b_max'],
                                        clip=config['data']['preprocessing'][img_key]['clip'])
                                    )

        elif img_key in config['data']['preprocessing']['needs_label_mapping']: # Remapping of segmentaion data
            preproc_transforms.append(MapLabelValued(
                                        keys=[img_key],
                                        orig_labels=[index for index, x in enumerate(config['data']['preprocessing'][img_key]['target_labels'])],
                                        target_labels=config['data']['preprocessing'][img_key]['target_labels']) 
                ) 
    
    return preproc_transforms