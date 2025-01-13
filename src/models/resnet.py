"""
Daniel: A ResNet model with LeakyReLU activation function. (LReLU with alpha = 0 is equivalent to ReLU)
        As in ResNet, batchnorm is used (not instance norm as in Hung's ResNet_DCNN_LReLU model).
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, lrelu_alpha, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, lrelu_alpha, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        print("block", out.shape, x.shape)

        if self.downsample is not None:
            residual = self.downsample(x)

        #print(out.shape, residual.shape)
        out += residual
        out = self.act(out)

        return out


class ResNet_LReLU(nn.Module):

    def __init__(self,
                 config,
                 block,
                 layers,
                 block_inplanes,
                 lrelu_alpha,
                 use_bias=False,
                 n_input_channels=3,
                 #conv1_t_size=5,
                 #conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()


        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        conv1_kernel_size = config['model']['resnet']['conv1_kernel_size']
        conv1_stride = config['model']['resnet']['conv1_stride']

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_kernel_size, conv1_kernel_size, conv1_kernel_size),
                               stride=(conv1_stride, conv1_stride, conv1_stride),
                               padding=(conv1_kernel_size // 2, conv1_kernel_size // 2, conv1_kernel_size // 2),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.act = nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.backbone = torch.nn.ModuleList()
        for idx in range(len(block_inplanes)):
            stride = 2 if idx > 0 else 1
            layer = self._make_layer(block, block_inplanes[idx], layers[idx],
                                       shortcut_type, lrelu_alpha, stride=stride)
            
            self.backbone.add_module(f'block%s' % idx, layer)
            

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        #flatten_layer_size = block_inplanes[3] * block.expansion

        #self.output_head = MultiToxOutputHead(config=config, n_features=n_features)


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, lrelu_alpha, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  lrelu_alpha=lrelu_alpha,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, lrelu_alpha))

        return nn.Sequential(*layers)


    
    def forward(self, x, autoencoder=False): 
        """
        Does a forward pass of only the backbone.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        for layer in self.backbone:
            x = layer(x)

        if autoencoder == False:
            x = self.avgpool(x)
        # x = self.avgpool(x)

        #x = self.flatten(x)

        return x
    


def get_resnet(config : dict, model_depth : int, channels : int, lrelu_alpha : float, **kwargs):
    """
    ResNet

    Args:
        config (dict): config params
        model_depth (int): ResNet model depth
        channels (int): number of input channels
        lrelu_alpha (float): leaky ReLU value
        **kwargs:

    Returns:
        model: a ResNet model

    """
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    # assert len(filters) == 4 or len(filters) == 3

    if model_depth == 10:
        model = ResNet_LReLU(config, BasicBlock, [1, 1, 1, 1], block_inplanes=[64, 128, 192, 256], n_input_channels=channels, 
                             lrelu_alpha=lrelu_alpha, **kwargs)
    elif model_depth == 18:
        model = ResNet_LReLU(config, BasicBlock, [2, 2, 2, 2], block_inplanes=[64, 128, 256, 512], n_input_channels=channels, 
                            lrelu_alpha=lrelu_alpha, **kwargs)
    elif model_depth == 34:
        model = ResNet_LReLU(config, BasicBlock, [3, 4, 6, 3], block_inplanes=[64, 128, 256, 512], n_input_channels=channels, 
                             lrelu_alpha=lrelu_alpha, **kwargs)
    elif model_depth == 50:
        model = ResNet_LReLU(config, Bottleneck, [3, 4, 6, 3], block_inplanes=[64, 128, 256, 512], n_input_channels=channels,
                             lrelu_alpha=lrelu_alpha, **kwargs)
    elif model_depth == 101:
        model = ResNet_LReLU(config, Bottleneck, [3, 4, 23, 3], block_inplanes=[64, 128, 256, 512], n_input_channels=channels, 
                             lrelu_alpha=lrelu_alpha, **kwargs)
    elif model_depth == 152:
        model = ResNet_LReLU(config, Bottleneck, [3, 8, 36, 3], block_inplanes=[64, 128, 256, 512], n_input_channels=channels, 
                             lrelu_alpha=lrelu_alpha, **kwargs)
    elif model_depth == 200:
        model = ResNet_LReLU(config, Bottleneck, [3, 24, 36, 3], block_inplanes=[64, 128, 256, 512], n_input_channels=channels, 
                             lrelu_alpha=lrelu_alpha, **kwargs)

    return model

# def get_model():
#     """
#     Initialize model.
#
#     Returns:
#
#     """
#     model = DenseNet121(
#         spatial_dims=config.spatial_dims,
#         in_channels=config.n_input_channels,
#         out_channels=config.num_classes,
#     )
#     return model.to(device=config.device)

