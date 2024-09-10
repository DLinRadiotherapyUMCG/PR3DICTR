"""
Same as DCNN, but where LeakyReLU activation has been applied whenever possible.
"""
import math
import torch
from src.models.conv_layers import conv3d_padding_same, pooling_conv
# from models.linear_layers import MultiToxOutputHead


class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 normalization='batch', pooling='max', pooling_size=2, use_bias=False):
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
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1,
                                      bias=use_bias)
        self.use_activation = use_activation
        self.activation2 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

        if pooling == None:
            self.pool = None
        elif pooling.lower() == 'max':
            self.pool = torch.nn.MaxPool3d(kernel_size=(pooling_size, pooling_size, pooling_size))
        elif pooling.lower() == 'avg' or 'average':
            self.pool = torch.nn.AvgPool3d(kernel_size=(pooling_size, pooling_size, pooling_size))
        elif pooling.lower() == 'none' or pooling == None:
            self.pool = None
        else:
            raise ValueError("Pooling is invalid:", pooling)
        
        if normalization == None:
            self.norm1 = None
            self.norm2 = None
        elif normalization.lower() == "instance":
            self.norm1 = torch.nn.InstanceNorm3d(filters)
            self.norm2 = torch.nn.InstanceNorm3d(filters)
        elif normalization.lower() == "batch":
            self.norm1 = torch.nn.BatchNorm3d(filters)
            self.norm2 = torch.nn.BatchNorm3d(filters)
        elif normalization.lower() == ("none" or None):
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError("Normalization is invalid:", normalization, "\n   CHOOSE: 'instance', 'batch' or None")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)

        x = self.norm1(x)
        
        if self.pool is not None:
            x = self.pool(x)

        if self.use_activation:
            x = self.activation1(x)

        x = self.pad(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_activation:
            x = self.activation2(x)

        return x



class small_conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation, normalization='batch',
                 pooling='max', pooling_size=2, use_bias=False):
        super(small_conv_block, self).__init__()

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
        
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        
        self.use_activation = use_activation

        if normalization == None:
            self.norm1 = None
            self.norm2 = None
        elif normalization.lower() == "instance":
            self.norm1 = torch.nn.InstanceNorm3d(filters)
        elif normalization.lower() == "batch":
            self.norm1 = torch.nn.BatchNorm3d(filters)
        elif normalization.lower() == ("none" or None):
            self.norm1 = None
        else:
            raise ValueError("Normalization is invalid:", normalization, "\n   CHOOSE: 'instance', 'batch' or None")
        
        if pooling == None:
            self.pool = None
        elif pooling.lower() == 'max':
            self.pool = torch.nn.MaxPool3d(kernel_size=(pooling_size, pooling_size, pooling_size))
        elif pooling.lower() == 'avg' or 'average':
            self.pool = torch.nn.AvgPool3d(kernel_size=(pooling_size, pooling_size, pooling_size))
        elif pooling.lower() == 'none' or pooling == None:
            self.pool = None
        else:
            raise ValueError("Pooling is invalid:", pooling)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)

        x = self.norm1(x)
        
        if self.pool is not None:
            x = self.pool(x)

        if self.use_activation:
            x = self.activation1(x)

        return x


class DCNN_Pooling(torch.nn.Module):
    """
    Deep CNN that used pooling (and not stride) to reduce the feature map size.
    """

    def __init__(self, config,
                 n_input_channels, n_features, filters,
                 kernel_sizes, strides, pad_value, lrelu_alpha, 
                 pooling,                 
                 use_bias=False):
        super(DCNN_Pooling, self).__init__()
        self.n_features = n_features

        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        self.first_conv_blocks = torch.nn.ModuleList()
        self.conv_blocks = torch.nn.ModuleList()
        pooling_size = config['model']['pooling_size']
        normalization = config['model']['normalization']

        if config['model']['conv_block_layers'] == 1:
            for i in range(len(in_channels)):
                use_activation = True
                self.conv_blocks.add_module('conv_block%s' % i,
                                            small_conv_block(in_channels=in_channels[i], filters=filters[i],
                                                    kernel_size=kernel_sizes[i], strides=strides[i],
                                                    pad_value=pad_value, lrelu_alpha=lrelu_alpha,
                                                    use_activation=use_activation, normalization=normalization, 
                                                    pooling=pooling, pooling_size=pooling_size, use_bias=use_bias))
        else:
            for i in range(len(in_channels)):
                use_activation = True
                self.conv_blocks.add_module('conv_block%s' % i,
                                            conv_block(in_channels=in_channels[i], filters=filters[i],
                                                    kernel_size=kernel_sizes[i], strides=strides[i],
                                                    pad_value=pad_value, lrelu_alpha=lrelu_alpha, 
                                                    use_activation=use_activation, normalization=normalization,
                                                    pooling=pooling, pooling_size=pooling_size,  use_bias=use_bias))

        # if config['model']['conv_block_layers'] == 1:
        #     for i in range(len(in_channels)):
        #         use_activation = True
        #         if i == 0:
        #             for j in range(in_channels[i]):
        #                 self.first_conv_blocks.add_module('conv_block1_%s' % j, small_conv_block(in_channels=1, filters=filters[i],
        #                                                     kernel_size=kernel_sizes[i], strides=strides[i],
        #                                                     pad_value=pad_value, lrelu_alpha=lrelu_alpha,
        #                                                     use_activation=use_activation, normalization=normalization, 
        #                                                     pooling=pooling, pooling_size=pooling_size, use_bias=use_bias))
        #         else:
        #             self.conv_blocks.add_module('conv_block%s' % i,
        #                                         small_conv_block(in_channels=in_channels[i], filters=filters[i],
        #                                                 kernel_size=kernel_sizes[i], strides=strides[i],
        #                                                 pad_value=pad_value, lrelu_alpha=lrelu_alpha,
        #                                                 use_activation=use_activation, normalization=normalization, 
        #                                                 pooling=pooling, pooling_size=pooling_size, use_bias=use_bias))
        # else:
        #     for i in range(len(in_channels)):
        #         use_activation = True
        #         if i == 0:
        #             for j in range(in_channels[i]):
        #                 self.first_conv_blocks.add_module('conv_block1_%s' % j, conv_block(in_channels=1, filters=filters[i],
        #                                                 kernel_size=kernel_sizes[i], strides=strides[i],
        #                                                 pad_value=pad_value, lrelu_alpha=lrelu_alpha, 
        #                                                 use_activation=use_activation, normalization=normalization,
        #                                                 pooling=pooling, pooling_size=pooling_size,  use_bias=use_bias))
        #         else:
        #             self.conv_blocks.add_module('conv_block%s' % i,
        #                                     conv_block(in_channels=in_channels[i], filters=filters[i],
        #                                             kernel_size=kernel_sizes[i], strides=strides[i],
        #                                             pad_value=pad_value, lrelu_alpha=lrelu_alpha, 
        #                                             use_activation=use_activation, normalization=normalization,
        #                                             pooling=pooling, pooling_size=pooling_size,  use_bias=use_bias))


    def forward(self, x, autoencoder=False):
        feature_maps = None
        # for idx, block in enumerate(self.first_conv_blocks):
        #     #print(idx)
        #     xtemp = x[:,idx,:,:,:].unsqueeze(1)
        #     if feature_maps is None:
        #         #print(xtemp.shape)
        #         feature_maps = block(xtemp)
        #         #print(feature_maps.shape)
        #     else:
        #         feature_maps += block(xtemp)
        # x = feature_maps

        for block in self.conv_blocks:
            x = block(x)

        return x
    
        


