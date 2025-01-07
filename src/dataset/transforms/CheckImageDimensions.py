import numpy as np

from monai.transforms import MapTransform
from monai.utils.type_conversion import  convert_to_tensor
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union, Mapping, Optional, Any
DtypeLike = Union[np.dtype, type, str, None]
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.enums import TransformBackends




class CheckImageDimensions(MapTransform):
    """
    Checks if the image data has the correct number of dimensions. If not, it adds a dimension at the start.

    Example: if we want CT, RTDose and RTSTruct to be concatenated into [batch_size, 3, 96, 96, 96], then each image
        should be [1,96,96,96]. This transform checks if it is correct. If the first dimension is missing (e.g. the image files are [96,96,96]),
        then it adds the first dimension using pytorch unsqueeze(0). 
    """
    backend = MapTransform.backend

    def __init__(
        self, keys, desired_num_img_dims=3,  # 3D images
    ) -> None:
        super().__init__(keys)

        self.desired_num_img_dims = desired_num_img_dims
        


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        #print(d)
        for key in self.key_iterator(d):
            
            num_dims = len(d[key].shape)
            if num_dims == self.desired_num_img_dims + 1:
                pass

            elif num_dims == self.desired_num_img_dims:   # if we're just missing the first dimension, its probably just because we're missing the first dimension (where the images are concatenated)
                d[key] = d[key].unsqueeze(0)             # so we add a dimension at the start

            else:
                raise ValueError(f"Image has {num_dims} dimensions, but expected {self.desired_num_img_dims} dimensions.")
            pass
            
        return d
    