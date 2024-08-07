"""
Same as DCNN, but where LeakyReLU activation has been applied whenever possible.
"""
import math
import torch
from functools import reduce
from operator import __add__
import numpy as np

from models.conv_layers import conv3d_padding_same
from models.linear_layers import MultiToxOutputHead

class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 normalization="instance", use_bias=False):
        super(conv_block, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)

        self.pad = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=pad_value)
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        #self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1,
                                     bias=use_bias)
        #self.norm2 = torch.nn.InstanceNorm3d(filters)
        self.use_activation = use_activation
        self.activation2 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

        if normalization == "instance":
            self.norm1 = torch.nn.InstanceNorm3d(filters)
            self.norm2 = torch.nn.InstanceNorm3d(filters)
        elif normalization == "batch":
            self.norm1 = torch.nn.BatchNorm3d(filters)
            self.norm2 = torch.nn.BatchNorm3d(filters)
        elif normalization == ("none" or None):
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError("Normalization is invalid:", normalization, "\n   CHOOSE: 'instance', 'batch' or None")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)

        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.activation1(x)
        x = self.pad(x)

        x = self.conv2(x)
        if self.norm2 is not None:
            x = self.norm2(x)

        if self.use_activation:
            x = self.activation2(x)
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



class DCNN_LReLU(torch.nn.Module):
    """
    Deep CNN
    """
    def __init__(self, config,
                 n_input_channels, depth, height, width,  filters,
                 kernel_sizes, strides, pad_value, n_down_blocks, lrelu_alpha, pooling_conv_filters,
                 perform_pooling, n_features,
                 
                 # linear_units, dropout_p, linear_units_endpoint, num_ohe_classes
                 # endpoint_list, n_features, clinical_variables_position, clinical_variables_linear_units
                 use_bias=False):
        super(DCNN_LReLU, self).__init__()
        
        
        self.pooling_conv_filters = pooling_conv_filters
        normalization = config.normalization
        self.perform_pooling = perform_pooling               

        # Determine Linear input channel size
        end_depth = math.ceil(depth / (2 ** n_down_blocks[0]))
        end_height = math.ceil(height / (2 ** n_down_blocks[1]))
        end_width = math.ceil(width / (2 ** n_down_blocks[2]))

        # ----- SHARED LAYERS ----- #
        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        self.blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):
            use_activation = True
            self.blocks.add_module('ConvBlock_shared_%s' % i,
                                        conv_block(in_channels=in_channels[i], filters=filters[i],
                                                   kernel_size=kernel_sizes[i], strides=strides[i],
                                                   pad_value=pad_value, lrelu_alpha=lrelu_alpha,
                                                   use_activation=use_activation, normalization=normalization,
                                                 use_bias=use_bias))

        # Initialize pooling conv
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
            # self.pool = torch.nn.AvgPool3d(kernel_size=(end_depth, end_height, end_width))
            self.pool = torch.nn.AvgPool3d(kernel_size=(end_depth, end_height, end_width))
            end_depth, end_height, end_width = 1, 1, 1


        # Initialize linear layers
        # Flatten layer (size can be different due to different backbone architectures, but the MultiToxOutputHead can handle this)
        self.flatten = torch.nn.Flatten()
        # make the linear layers (shared, clinical and non-shared)
        self.output_head = MultiToxOutputHead(config=config, n_features=n_features)
        


    def forward(self, x, features):
        # ----- SHARED LAYERS ----- #
        # Convolutional (SHARED)
        for block in self.blocks:
            x = block(x)

        # Pooling layers (SHARED)
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)

        x = self.flatten(x)

        # MLP clinical variables (SHARED)
        x_dict = self.output_head(x, features)

        return x_dict


