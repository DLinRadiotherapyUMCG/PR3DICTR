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

from monai.networks.layers.factories import Conv, Dropout, Pool



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
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class ModulatedDenseNet(nn.Module):
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
        config: dict,
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

        first_kernel_size = 5


        first_block = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=first_kernel_size, stride=(2, 2, 2), padding=first_kernel_size // 2, bias=False),
            nn.BatchNorm3d(init_features),
            nn.LeakyReLU(negative_slope=0),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
    
        self.features = nn.Sequential(
            
        )

        self.features.add_module('first_block', first_block)
        

        linear_layer_sizes = [init_features]

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

            linear_layer_sizes.append(in_channels)

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


        # LINEAR LAYER STUFF
        
        self.CLC_MLP = nn.Sequential()
        self.FiLM_layers = nn.Sequential()

        in_features = 1 # len(config['columns']['clinical_features']) 
        
        for i, out_features in enumerate(linear_layer_sizes): 
            self.CLC_MLP.add_module("CLC_MLP_layer_%d" % (i + 1),
                LinearLayerBlock(
                    in_features=in_features,
                    out_features=out_features,
                    act='selu',
                )
            )

            self.FiLM_layers.add_module(
                "FiLM_layer_%d" % (i + 1),
                FiLMSpatialModulation3D()
            )

            in_features = out_features

        self.modulation_block = ClinicalModulation()
            

    def forward(self, x: torch.Tensor, features: torch.Tensor = None,  autoencoder=False) -> torch.Tensor:
        
        clc_layers_idx = 0

        for idx, layer in enumerate(self.features): # DenseNet blocks
            x = layer(x)

            if (idx == 0) or (idx % 2 == 1):  # after the very first block, and every dense block
                features = self.CLC_MLP[clc_layers_idx](features)
                x = self.FiLM_layers[clc_layers_idx](x, features)

                clc_layers_idx += 1

                
            

        if autoencoder == False:
            x = self.avgpool(x)
       
        return x


class LinearLayerBlock(nn.Module):
    """
    Linear layer with activation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act: Union[str, tuple] = ("selu", {"inplace": True}),  # the authors used selu
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)

        if self.act is not None:
            x = self.act(x)

        return x



class ClinicalModulation(nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = nn.Linear(clinical_dim, in_channels)

    def forward(self, x, clinical_vector):
        
        return x * clinical_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # broadcast over spatial dims




class FiLMSpatialModulation3D(nn.Module):
    def __init__(self):
        super().__init__()
        #self.num_channels = None
        #self.spatial_shape = None  # Tuple: (D, H, W)
        #self.clinical_dim = clinical_dim
        
        self.spatial_gate = None

    def init_spatial_gate(self):
        # FiLM parameters (scale and shift)
        B, C, D, H, W = self.spatial_shape

        
        self.FiLM_layer = nn.Linear(self.clinical_dim, C * 2)

        # Spatial attention (gating)
        #D, H, W = spatial_shape
        self.spatial_gate = nn.Sequential(
            nn.Linear(self.clinical_dim, D * H * W),
            nn.Sigmoid()
        )

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x, clinical_vector):
        shape = x.shape

        if self.spatial_gate is None:
            self.clinical_dim = clinical_vector.shape[-1]
            self.spatial_shape = shape
            self.init_spatial_gate()  # Initialize spatial gate with the shape of x

        # x: [B, C, D, H, W], clinical: [B, clinical_dim]
        B, C, D, H, W = shape

        # FiLM modulation
        film_params = self.FiLM_layer(clinical_vector)  # [B, 2C]
        gamma, beta = film_params.chunk(2, dim=1)  # Each [B, C]
        gamma = gamma.view(B, C, 1, 1, 1)
        beta = beta.view(B, C, 1, 1, 1)
        x = gamma * x + beta

        # Spatial gating
        gate = self.spatial_gate(clinical_vector).view(B, 1, D, H, W)  # [B, 1, D, H, W]
        x = x * gate  # element-wise spatial mask

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


def get_modulated_densenet(config : dict, model_depth : int, channels: int):
    """
    Loads a DenseNet model.

    Args:
        config:
        model_depth:
        channels:

    Returns:
        model: a DenseNet backbone
    """
    
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4)
    elif model_depth == 169:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), bn_size=4)
    elif model_depth == 201:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), bn_size=4)
    elif model_depth == 264:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), bn_size=4)
    

    return model


def get_short_modulated_densenet(config : dict, model_depth : int, channels: int):
    """
    Loads a shorter DenseNet model. It is identical to the regular densenet, but is missing the fourth block. This is useful
    for models such as TransRP, which may want to retain a higher resoultion feature map at the end of the DenseNet block

    Args:
        config:
        model_depth:
        channels:

    Returns:
        model: a DenseNet backbone
    """

    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 24), bn_size=4)
    elif model_depth == 169:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 32), bn_size=4)
    elif model_depth == 201:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 48), bn_size=4)
    elif model_depth == 264:
        model = ModulatedDenseNet(config, in_channels=channels, init_features=64, growth_rate=32, block_config=(6, 12, 64), bn_size=4)
    

    return model