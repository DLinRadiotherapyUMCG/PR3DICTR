import torch
from monai.transforms import (
    Transform,
    MapTransform,
    RandomizableTransform,
    Randomizable,
    Compose,
    Rand3DElastic,
    RandAdjustContrast,
    RandFlip,
    RandGaussianNoise,
    RandAffine,
    RandRotate,
    RandShiftIntensity,
    ScaleIntensityRange,
    NormalizeIntensity,
)

from abc import abstractmethod

class Mixer(RandomizableTransform):

    def __init__(self, batch_size: int, alpha: float = 1.0) -> None:
        """
        Mixer is a base class providing the basic logic for the mixup-class of
        augmentations. In all cases, we need to sample the mixing weights for each
        sample (lambda in the notation used in the papers). Also, pairs of samples
        being mixed are picked by randomly shuffling the batch samples.

        Args:
            batch_size (int): number of samples per batch. That is, samples are expected tp
                be of size batchsize x channels [x depth] x height x width.
            alpha (float, optional): mixing weights are sampled from the Beta(alpha, alpha)
                distribution. Defaults to 1.0, the uniform distribution.
        """
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"Expected positive number, but got {alpha = }")
        self.alpha = alpha
        self.batch_size = batch_size

    @abstractmethod
    def apply(self, data: torch.Tensor):
        raise NotImplementedError()

    def randomize(self, data=None) -> None:
        """
        Sometimes you need may to apply the same transform to different tensors.
        The idea is to get a sample and then apply it with apply() as often
        as needed. You need to call this method everytime you apply the transform to a new
        batch.
        """
        self._params = (
            torch.from_numpy(self.R.beta(self.alpha, self.alpha, self.batch_size)).type(torch.float32),
            self.R.permutation(self.batch_size),
        )

""" Torch implementation """
# class MixUp(Randomizable, MapTransform):

#     backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

#     def __init__(self, keys, alpha, num_classes) -> None:
#         super(Randomizable, self).__init__(keys)
#         self.keys = keys
#         self.alpha = alpha
#         self.num_classes = num_classes
        
 
#     def randomize(self, data: Optional[Any] = None) -> None:
#         random_seed = random.getrandbits(32)
#         self.seed = random_seed

#     def apply_mixup(self):
#         pass

#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
#         self.randomize()

#         data = dict(data)
#         key = self.keys
#         ws = torch.tensor(np.random.dirichlet([1] * self.mixture_width), dtype=torch.float32)
#         m = torch.tensor(np.random.beta(1, 1), dtype=torch.float32)

#         for key in self.key_iterator(data):
#             input_img = data[key]
#             #print(input_img.shape)
#             mix = torch.zeros_like(input_img)#, device=self.device)    
        
#         return data

class MixUp(Mixer):
    """MixUp as described in:
    Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
    mixup: Beyond Empirical Risk Minimization, ICLR 2018

    Class derived from :py:class:`monai.transforms.Mixer`. See corresponding
    documentation for details on the constructor parameters.
    """

    def apply(self, data: torch.Tensor):
        weight, perm = self._params
        nsamples, *dims = data.shape
        
        if len(weight) != nsamples:
            raise ValueError(f"Expected batch of size: {len(weight)}, but got {nsamples}")

        if len(dims) not in [1, 2, 3, 4]:
            raise ValueError("Unexpected number of dimensions")

        mixweight = weight[(Ellipsis,) + (None,) * len(dims)]

        return mixweight * data + (1 - mixweight) * data[perm, ...]


    def __call__(self, data: torch.Tensor, labels: torch.Tensor | None = None):
        self.randomize()
        if labels is None:
            return self.apply(data)
        return self.apply(data), self.apply(labels)



class MixUpd(MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.MixUp`.

    Notice that the mixup transformation will be the same for all entries
    for consistency, i.e. images and labels must be applied the same augmenation.
    """

    def __init__(
        self, keys, batch_size: int, alpha: float = 1.0, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mixup = MixUp(batch_size, alpha)


    def __call__(self, data):
        self.mixup.randomize()
        result = dict(data)
        for k in self.keys:
            result[k] = self.mixup.apply(data[k])
        return result
