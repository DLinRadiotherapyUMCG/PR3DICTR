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

import time
import logging

from src.constants import DEVICE

class MixUp():
    """
    A class to handle the MixUp augmentation for the model.
    Performs it in the collate function, as it is a batch-level augmentation (and not patient-level).
    """
    def __init__(self, config):
        self.config = config
        self.alpha = config['data']['augmentation']['mixup']['alpha']

        self.mean_time = 0
        self.iters = 0

    def __call__(self, batch):

        batch = list_data_collate(batch)

        keys = list(batch.keys())
        batch_size = batch[keys[0]].shape[0]

        # randomly samply a lambda value from a beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        # create a random permutation of the batch indices
        index = torch.randperm(batch_size, device=batch[keys[0]].device)

        # perform the mixup on the model inputs (labels are dealt with in the training code)
        for key in ['input', 'features']: # , 'label_list']:  # , 'features', 'label_list'
            data = batch[key]

            data.mul_(lam).add_((1 - lam) * data[index, :])

        # return the lambda value and mixing indicies with the batch
        batch['lambda'] = lam
        batch['indices'] = index
        
        return batch




