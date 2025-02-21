import numpy as np
import torch
from monai.transforms.regularization.array import Mixer
from monai.transforms import MapTransform, RandomizableTransform
from monai.utils.type_conversion import  convert_to_tensor
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union, Mapping, Optional, Any
DtypeLike = Union[np.dtype, type, str, None]
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.enums import TransformBackends
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor
from monai.config import KeysCollection
from monai.data.utils import list_data_collate



class MixUp():
    """
    A class to handle the MixUp augmentation for the model.
    Performs it in the collate function, as it is a batch-level augmentation (and not patient-level).
    """
    def __init__(self, config):
        self.config = config
        self.alpha = config['data']['augmentation']['mixup']['alpha']

    def __call__(self, batch):

        # first collate the batch
        batch = list_data_collate(batch)

        keys = list(batch.keys())
        batch_size = batch[keys[0]].shape[0]
        #device = batch[keys[0]].device
        
        # randomly samply a lambda value from a beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        # create a random permutation of the batch indices
        index = torch.randperm(batch_size)

        # perform the mixup on the model inputs (labels are dealt with in the training code)
        mixed_data = batch
        for key in ['input', 'features']: # , 'label_list']:  # , 'features', 'label_list'
            #print(key)
            mixed_data[key] = lam * batch[key] + (1 - lam) * batch[key][index, :]

        # return the lambda value and mixing indicies with the batch
        mixed_data['lambda'] = lam
        mixed_data['indices'] = index

        return mixed_data




