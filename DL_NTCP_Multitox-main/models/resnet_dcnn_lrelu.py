"""
Same as ResNet_DCNN, but where every ReLU activation has been replaced by LeakyReLU.
"""
import math
import torch
from functools import reduce
from operator import __add__
import numpy as np


from models.conv_layers import conv3d_padding_same
from models.linear_layers import MultiToxOutputHead



def conv3x3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, pad_value, lrelu_alpha, normalization='instance', stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        # self.conv1 = torch.nn.utils.parametrizations.spectral_norm(conv3x3x3(in_planes, planes, stride))
        # self.norm1 = torch.nn.BatchNorm3d(planes)
       # self.norm1 = torch.nn.InstanceNorm3d(planes)
        self.activation = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = conv3x3x3(planes, planes * self.expansion)
        # DWS
        # self.pad = conv3d_padding_same(depth=3, height=3, width=3, pad_value=pad_value)
        # self.conv2_1 = torch.nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=3,
        #                                stride=stride, bias=False, groups=planes)
        # self.conv2_2 = torch.nn.Conv3d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1,
        #                                stride=1, bias=False)

        # self.conv2 = torch.nn.utils.parametrizations.spectral_norm(conv3x3x3(interm_features, interm_features,
        #                                                                      stride))

        # self.conv2 = torch.nn.utils.parametrizations.spectral_norm(conv3x3x3(planes, planes * self.expansion))
        # self.norm2 = torch.nn.BatchNorm3d(planes * self.expansion)
        #self.norm2 = torch.nn.InstanceNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        if normalization == 'instance':
            self.norm1 = torch.nn.InstanceNorm3d(planes)
            self.norm2 = torch.nn.InstanceNorm3d(planes * self.expansion)
        elif normalization == 'batch':
            self.norm1 = torch.nn.BatchNorm3d(planes)
            self.norm2 = torch.nn.BatchNorm3d(planes * self.expansion)
        elif normalization == ('none' or None):
            self.norm1 = None
            self.norm2 = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        # DWS
        # out = self.pad(out)
        # out = self.conv2_1(out)
        # out = self.conv2_2(out)

        if self.norm2 is not None:
            out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class InvertedResidual(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, pad_value, lrelu_alpha, stride=1, downsample=None):
        super().__init__()
        interm_features = planes * self.expansion

        self.conv1 = conv1x1x1(in_planes, interm_features)
        # self.conv1 = torch.nn.utils.parametrizations.spectral_norm(conv1x1x1(in_planes, interm_features))
        # self.norm1 = torch.nn.BatchNorm3d(interm_features)
        self.norm1 = torch.nn.InstanceNorm3d(interm_features)
        self.conv2 = conv3x3x3(interm_features, interm_features, stride)
        # DWS
        # self.conv2_1 = torch.nn.Conv3d(in_channels=interm_features, out_channels=interm_features, kernel_size=3,
        #                                stride=stride, bias=False, groups=interm_features)
        # self.conv2_2 = torch.nn.Conv3d(in_channels=interm_features, out_channels=interm_features, kernel_size=1,
        #                                stride=1, bias=False)

        # self.conv2 = torch.nn.utils.parametrizations.spectral_norm(conv3x3x3(interm_features, interm_features,
        #                                                                      stride))
        # self.norm2 = torch.nn.BatchNorm3d(interm_features)
        self.norm2 = torch.nn.InstanceNorm3d(interm_features)
        self.conv3 = conv1x1x1(interm_features, planes)
        # self.conv3 = torch.nn.utils.parametrizations.spectral_norm(conv1x1x1(interm_features, planes))
        # self.norm3 = torch.nn.BatchNorm3d(planes)
        self.norm3 = torch.nn.InstanceNorm3d(planes)
        self.activation = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        # DWS
        # out = self.conv2_1(out)
        # out = self.conv2_2(out)

        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 use_bias=False):
        super(conv_block, self).__init__()

        
        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int or np.issubdtype(type(kernel_size), int):
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size, type(kernel_size))

        self.pad = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=pad_value)
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        # DWS
        # self.conv1_1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
        #                                stride=strides, bias=use_bias, groups=in_channels)
        # self.conv1_2 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=1,
        #                                stride=1, bias=use_bias)

        # self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.use_activation = use_activation
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        # DWS
        # x = self.conv1_1(x)
        # x = self.conv1_2(x)

        # x = self.norm1(x)
        if self.use_activation:
            x = self.activation1(x)
        return x


class pooling_conv(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, lrelu_alpha, use_bias=False):
        super(pooling_conv, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        return x


class ResNet_DCNN_LReLU(torch.nn.Module):
    """
    ResNet + DCNN
    """
    def __init__(self, config,
                 n_input_channels, depth, height, width, filters,
                 kernel_sizes, strides, pad_value, n_down_blocks, lrelu_alpha, pooling_conv_filters,
                 perform_pooling, use_bias=False):
        super(ResNet_DCNN_LReLU, self).__init__()
        normalization = config.normalization
        self.pooling_conv_filters = pooling_conv_filters
        self.perform_pooling = perform_pooling
        

        # Determine Linear input channel size
        end_depth = math.ceil(depth / (2 ** n_down_blocks[0]))
        end_height = math.ceil(height / (2 ** n_down_blocks[1]))
        end_width = math.ceil(width / (2 ** n_down_blocks[2]))

        #flatten_layer_size = end_depth * end_height * end_width * filters[-1]

        # ----- SHARED LAYERS ----- #
        # Initialize conv blocks (SHARED)
        in_channels = [n_input_channels] + list(filters[:-1])
        self.blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):
            use_activation = True
            # Residual block
            self.blocks.add_module('ResBlock_shared_%s' % i, BasicResBlock(in_planes=in_channels[i], planes=in_channels[i],
                                                                           pad_value=pad_value, lrelu_alpha=lrelu_alpha,
                                                                           normalization=normalization))
            # Downsampling conv block
            self.blocks.add_module('ConvBlock_shared_%s' % i,
                                   conv_block(in_channels=in_channels[i], filters=filters[i],
                                              kernel_size=kernel_sizes[i], strides=strides[i], pad_value=pad_value,
                                              lrelu_alpha=lrelu_alpha, use_bias=use_bias,
                                              use_activation=use_activation))

        # Initialize pooling conv (SHARED)
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [end_depth, end_height, end_width]
            self.pool = pooling_conv(in_channels=filters[-1], filters=pooling_conv_filters,
                                     kernel_size=pooling_conv_kernel_size, strides=1,
                                     lrelu_alpha=lrelu_alpha, use_bias=use_bias)
            end_depth, end_height, end_width = 1, 1, 1
            filters[-1] = self.pooling_conv_filters
        elif self.perform_pooling:
            # self.pool = torch.nn.AvgPool3d(kernel_size=(1, end_height, end_width))
            # end_depth, end_height, end_width = depth, 1, 1
            self.pool = torch.nn.AvgPool3d(kernel_size=(end_depth, end_height, end_width))
            # self.pool = torch.nn.MaxPool3d(kernel_size=(end_depth, end_height, end_width))
            end_depth, end_height, end_width = 1, 1, 1


        # Initialize linear layers
        #self.flatten = torch.nn.Flatten()

        #self.output_head = MultiToxOutputHead(config=config, n_features=n_features)
        


    def forward(self, x, autoencoder=False):
        # ----- SHARED LAYERS ----- #
        # Convolutional (SHARED)
        for block in self.blocks:
            x = block(x)
            #print(x.shape)

        # Pooling layers (SHARED)
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)


        return x


