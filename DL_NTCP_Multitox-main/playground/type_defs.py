from typing import TypedDict, List, Optional, Any, Callable, Dict, Iterable, Tuple, Union

from torch import Tensor


class ScaleIntensityRangeSettings(TypedDict):
    a_min: float
    a_max: float
    b_min: float
    b_max: float
    clip: bool

class SegmentationMapSettings(TypedDict):
    target_labels: List[float]

class Preprocessing(TypedDict):
    crop: bool
    crop_shape: List[int]
    ct: ScaleIntensityRangeSettings
    rtdose: ScaleIntensityRangeSettings
    segmentation_map: SegmentationMapSettings



class ModelConfig(TypedDict):
    model_name: str
    filters: List[int]


class Config(TypedDict):
    seed: int
    #preprocessing: Preprocessing
    model: ModelConfig
    