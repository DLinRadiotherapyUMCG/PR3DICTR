import numpy as np

from monai.transforms import MapTransform
from monai.utils.type_conversion import  convert_to_tensor
from typing import Collection, Hashable, Iterable, Sequence, TypeVar, Union, Mapping, Optional, Any
DtypeLike = Union[np.dtype, type, str, None]
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.enums import TransformBackends




class ConvertMetaTensorToTensor(MapTransform):
    """
    Convert the meta data of the tensor to a pytorch tensor.
    """
    backend = MapTransform.backend

    def __init__(
        self, keys,
    ) -> None:
        super().__init__(keys)
        #self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = convert_to_tensor(d[key], track_meta=False)
            
        return d
    