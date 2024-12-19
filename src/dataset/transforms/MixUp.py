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


# class MixUp(MapTransform):
#     """
#     Convert the meta data of the tensor to a pytorch tensor.
#     """
#     backend = MapTransform.backend

#     def __init__(
#         self, keys,
#     ) -> None:
#         super().__init__(keys)
#         #self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip, dtype)

#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         for key in self.key_iterator(d):
#             d[key] = convert_to_tensor(d[key], track_meta=False)
            
#         return d
    

############################################################################################################

# class MONAI_MixUp(Mixer):
#     """MixUp as described in:
#     Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
#     mixup: Beyond Empirical Risk Minimization, ICLR 2018

#     Class derived from :py:class:`monai.transforms.Mixer`. See corresponding
#     documentation for details on the constructor parameters.
#     """

#     def apply(self, data: torch.Tensor):
#         weight, perm, _ = self._params
#         nsamples, *dims = data.shape
#         if len(weight) != nsamples:
#             #self.batch_size = nsamples
#             #self.randomize(None)
#             raise ValueError(f"Expected batch of size: {len(weight)}, but got {nsamples}")

#         if len(dims) not in [3, 4]:
#             raise ValueError("Unexpected number of dimensions")

#         mixweight = weight[(Ellipsis,) + (None,) * len(dims)]
#         #print(mixweight)
#         return mixweight * data + (1 - mixweight) * data[perm, ...]

#     def __call__(self, data: torch.Tensor, labels: torch.Tensor | None = None, randomize=True):
#         data_t = convert_to_tensor(data, track_meta=get_track_meta())
#         labels_t = data_t  # will not stay this value, needed to satisfy pylint/mypy

#         nsamples, *dims = data.shape
#         print(data.shape)
        
#         self.batch_size = nsamples
#             #self.randomize(None)

#         if labels is not None:
#             labels_t = convert_to_tensor(labels, track_meta=get_track_meta())
#         if randomize:
#             self.randomize()
#         if labels is None:
#             return convert_to_dst_type(self.apply(data_t), dst=data)[0]

#         return (
#             convert_to_dst_type(self.apply(data_t), dst=data)[0],
#             convert_to_dst_type(self.apply(labels_t), dst=labels)[0],
#         )


# class MONAI_MixUpd(MapTransform, RandomizableTransform):
#     """
#     Dictionary-based version :py:class:`monai.transforms.MixUp`.

#     Notice that the mixup transformation will be the same for all entries
#     for consistency, i.e. images and labels must be applied the same augmenation.
#     """

#     def __init__(
#         self, keys: KeysCollection, batch_size: int, alpha: float = 1.0, allow_missing_keys: bool = False
#     ) -> None:
#         MapTransform.__init__(self, keys, allow_missing_keys)
#         self.mixup = MONAI_MixUp(batch_size, alpha)

#     def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
#         super().set_random_state(seed, state)
#         self.mixup.set_random_state(seed, state)
#         return self

#     def __call__(self, data):
#         d = dict(data)
#         # all the keys share the same random state
#         self.mixup.randomize(None)
#         for k in self.key_iterator(d):
#             d[k] = self.mixup(data[k], randomize=False)
#         return d


############################################################################################################






class MixUp():
    def __init__(self, config):
        self.config = config
        self.alpha = 3.0

    def __call__(self, batch):

        # first collate the batch
        batch = list_data_collate(batch)

        keys = list(batch.keys())
        
        batch_size = batch[keys[0]].shape[0]
        device = batch[keys[0]].device

        #print(keys, batch_size)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        #device = data.device
        index = torch.randperm(batch_size)

        mixed_data = batch

        for key in ['input', 'features']: # , 'label_list']:  # , 'features', 'label_list'
            #print(key)
            mixed_data[key] = lam * batch[key] + (1 - lam) * batch[key][index, :]

            # if key == 'label_list':
            #     print(mixed_data[key])

        mixed_data['lambda'] = lam
        mixed_data['indices'] = index

        return mixed_data




