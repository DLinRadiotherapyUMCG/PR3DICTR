# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from timm.models.registry import register_model



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C)  # NOTE: now (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)  
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W)   # NOTE: now  (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x




class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x




class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, config, in_chans=3, drop_first_block=False,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        kernel_size = config['model']['convnext']['kernel_size']
        patch_size = config['model']['convnext']['patch_size']

        if drop_first_block:
            drop_first_block = depths[1:]
        #in_chans = channels
        self.num_blocks = len(depths)
        # print(depths)
        # print(self.num_blocks)
        # print(dims)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=patch_size, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_blocks-1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_blocks):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size=kernel_size,
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[self.num_blocks-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        #self.head.weight.data.mul_(head_init_scale)
        #self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)) and not isinstance(m, nn.LazyLinear):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, autoencoder=False):
        """
        Forward pass of the backbone
        """
    
        for i in range(self.num_blocks):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        if not autoencoder:
            #print(x.shape)
            return self.norm(x.mean([-3, -2, -1]))  # global average pooling, (N, C, H, W, D) -> (N, C)
        else:
            #print(x.shape)
            return x  # return feature maps
        





def get_convnext(config: dict, model_size: str, channels: int):
    
    model_size = model_size.lower()

    assert model_size in ["tiny", "small", "base", "large", "xlarge"], "model size must be one of 'tiny', 'small', 'base', 'large', 'xlarge'"
    
    if model_size == "tiny":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    elif model_size == "small":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
    elif model_size == "base":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    elif model_size == "large":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    elif model_size == "xlarge":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
    return model


def get_short_convnext(config: dict, model_size: str, channels: int):
    assert model_size in ["tiny", "small", "base", "large", "xlarge"], "model size must be one of 'tiny', 'small', 'base', 'large', 'xlarge'"
    if model_size == "tiny":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 9], dims=[96, 192, 384])
    elif model_size == "small":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27], dims=[96, 192, 384])
    elif model_size == "base":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27], dims=[128, 256, 512])
    elif model_size == "large":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27], dims=[192, 384, 768])
    elif model_size == "xlarge":
        model = ConvNeXt(config, in_chans=channels, depths=[3, 3, 27], dims=[256, 512, 1024])
    return model