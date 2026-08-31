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
    image_modality_config = config['data']['preprocessing']['image_modalities']  # get the (sub-)config for the different image modalities (e.g. CT, PET, MRI, etc.), just to shorten the code below

    # Goes through the different images and sets the required transform per volume types based on the input
    for img_key in image_keys:
        if img_key in config['data']['preprocessing']['needs_scaling']: # Any windowing and rescaling for the input
            preproc_transforms.append(ScaleIntensityRanged(keys=[img_key],
                                        a_min=image_modality_config[img_key]['a_min'], a_max=image_modality_config[img_key]['a_max'],                                         # NOTE: commented out as segmap is not used
                                        b_min=image_modality_config[img_key]['b_min'], b_max=image_modality_config[img_key]['b_max'],
                                        clip=image_modality_config[img_key]['clip'])
                                    )

        elif img_key in config['data']['preprocessing']['needs_label_mapping']: # Remapping of segmentaion data
            preproc_transforms.append(MapLabelValued(
                                        keys=[img_key],
                                        orig_labels=[index for index, x in enumerate(image_modality_config[img_key]['target_labels'])],
                                        target_labels=image_modality_config[img_key]['target_labels']) 
                ) 
    
    return preproc_transforms