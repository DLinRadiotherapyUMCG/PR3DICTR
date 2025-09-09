import math
import torch
from src.models.conv_layers import conv3d_padding_same, pooling_conv


class conv_block(torch.nn.Module):
    """
    A CNN block with two convolutional layers
    """
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
    """
    A CNN block with just 1 convolutional layer
    """
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










class CNN_Pooling(torch.nn.Module):
    """
    Deep CNN that used pooling (and not stride) to reduce the feature map size.
    """

    def __init__(self, config, n_input_channels):
        super(CNN_Pooling, self).__init__()
        
        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        self.first_conv_blocks = torch.nn.ModuleList()
        self.conv_blocks = torch.nn.ModuleList()

        filters = config['model']['cnn_pooling']['filters']
        kernel_sizes = config['model']['cnn_pooling']['kernel_sizes']
        strides = config['model']['cnn_pooling']['strides']
        pooling = config['model']['cnn_pooling']['pooling']
        pooling_size = config['model']['cnn_pooling']['pooling_size']
        normalization = config['model']['cnn_pooling']['normalization']
        pad_value = config['model']['cnn_pooling']['pad_value']
        lrelu_alpha = config['model']['lrelu_alpha']
        use_bias = use_bias = config['model']['use_bias']  # usually false, as we use normalization layers

        if config['model']['cnn_pooling']['conv_block_layers'] == 1: 
            block_module = small_conv_block
        else:
            block_module = conv_block

        for i in range(len(in_channels)):
            use_activation = True
            self.conv_blocks.add_module('conv_block_%s' % i,
                                            block_module(in_channels=in_channels[i], filters=filters[i],
                                                    kernel_size=kernel_sizes[i], strides=strides[i],
                                                    pad_value=pad_value, lrelu_alpha=lrelu_alpha,
                                                    use_activation=use_activation, normalization=normalization, 
                                                    pooling=pooling, pooling_size=pooling_size, use_bias=use_bias))

        
        # TODO: add average pooling to CNN
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)


    def forward(self, x, autoencoder=False):

        for block in self.conv_blocks:
            x = block(x)

        if autoencoder == False:
            x = self.avgpool(x)

        return x
    
        


