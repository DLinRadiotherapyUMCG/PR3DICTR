# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option


class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", nn.BatchNorm3d(in_channels))
        self.layers.add_module("relu1", nn.LeakyReLU(negative_slope=0))
        self.layers.add_module("conv1",  nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", nn.BatchNorm3d(out_channels))
        self.layers.add_module("relu2", nn.LeakyReLU(negative_slope=0))
        self.layers.add_module("conv2",  nn.Conv3d(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        #conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        #pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", nn.BatchNorm3d(in_channels))
        self.add_module("relu", nn.LeakyReLU(negative_slope=0))
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", nn.MaxPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-deterministic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        n_input_channels = in_channels

        conv_type = nn.Conv3d
        pool_type = nn.MaxPool3d
        avg_pool_type = nn.AdaptiveAvgPool3d

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(in_channels, init_features, kernel_size=7, stride=(2,4,4), padding=2, bias=False)),
                    ("norm0", nn.BatchNorm3d(init_features)),
                    ("relu0", nn.LeakyReLU(negative_slope=0)),
                    ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=n_input_channels,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock_{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", nn.BatchNorm3d(in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    n_input_channels, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition_{i + 1}", trans)
                in_channels = _out_channels
       

        # pooling and chang to 64 features
        # self.class_layers = nn.Sequential(
        #     OrderedDict(
        #         [
        #             ("relu", get_act_layer(name=act)),
        #             ("pool", avg_pool_type(1)),
        #         ]
        #     )
        # )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

        # try:
        #self.features.transition3.pool.stride  = 1  

        self.avgpool = avg_pool_type(1)

        

    def forward(self, x: torch.Tensor, autoencoder=False) -> torch.Tensor:
        for layer in self.features:
            #print(layer)
            x = layer(x)

        if autoencoder == False:
            x = self.avgpool(x)
       
        return x


"""

in_channels: int = 3,
init_features: int = 64,
growth_rate: int = 32,
block_config: Sequence[int] = (6, 12, 24, 16),
bn_size: int = 4,
act: Union[str, tuple] = ("relu", {"inplace": True}),
norm: Union[str, tuple] = "batch",
dropout_prob: float = 0.0,
"""


def get_desnsenet(config, model_depth, channels):
    """
    DenseNet

    Args:
        config:
        model_depth:
        channels:
        n_features:
        filters:
        lrelu_alpha:
        **kwargs:

    Returns:

    """
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4)
    elif model_depth == 169:
        model = DenseNet(in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), bn_size=4)
    elif model_depth == 201:
        model = DenseNet(in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), bn_size=4)
    elif model_depth == 264:
        model = DenseNet(in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), bn_size=4)
    

    return model
