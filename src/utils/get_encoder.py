# -*- coding: utf-8 -*-

from src.models.dcnn_pooling import DCNN_Pooling
from src.models.resnet import get_resnet
from src.models.densenet import get_desnsenet

def get_encoder(config, channels, depth, height, width, n_features):
    model_name = config['model']['model_name']
    filters = config['model']['filters']
    kernel_sizes = config['model']['kernel_sizes']
    strides = config['model']['strides']
    pad_value = config['model']['pad_value']
    lrelu_alpha = config['model']['lrelu_alpha']

    #dropout_p_endpoint = config.dropout_p_endpoint
    use_bias = config['model']['use_bias']

    # if config.siamese_encoder:
    #     channels = 1

    """
    Initialize model.
    """
    # Determine number of downsampling blocks
    n_down_blocks = [sum([x[0] == 2 for x in strides]), sum([x[1] == 2 for x in strides]),
                     sum([x[2] == 2 for x in strides])]

    if model_name == 'dcnn_pooling':
        encoder = DCNN_Pooling(config=config,
                                n_input_channels=channels,n_features=n_features, 
                                  filters=filters, kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                  lrelu_alpha=lrelu_alpha, pooling=config['model']['pooling'],
                                  use_bias=use_bias)  
    elif model_name == 'resnet':
        encoder = get_resnet(config=config, model_depth=config['model']['resnet']['model_depth'], channels=channels, lrelu_alpha=lrelu_alpha)
    
    elif model_name == 'densenet':
        encoder = get_desnsenet(config, config['model']['densenet']['model_depth'], channels)
    
    else:
        raise ValueError('Invalid model_name: {}.'.format(model_name))

    
    return encoder
