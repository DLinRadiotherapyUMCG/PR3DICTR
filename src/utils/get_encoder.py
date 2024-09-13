# -*- coding: utf-8 -*-

from src.models.dcnn_pooling import DCNN_Pooling
from src.models.resnet import get_resnet
from src.models.densenet import get_desnsenet
from src.models.ViT import ViT


def get_encoder(config, channels, depth, height, width, n_features):
    """
    Initialize the image encoder.
    """

    model_name = config['model']['model_name']
    filters = config['model']['filters']
    kernel_sizes = config['model']['kernel_sizes']
    strides = config['model']['strides']
    pad_value = config['model']['pad_value']
    lrelu_alpha = config['model']['lrelu_alpha']

    #dropout_p_endpoint = config.dropout_p_endpoint
    use_bias = config['model']['use_bias']

    # Determine number of downsampling blocks
    # n_down_blocks = [sum([x[0] == 2 for x in strides]), sum([x[1] == 2 for x in strides]),
    #                  sum([x[2] == 2 for x in strides])]

    if model_name == 'dcnn_pooling':
        encoder = DCNN_Pooling(config=config,
                                n_input_channels=channels, 
                                  filters=filters, kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                  lrelu_alpha=lrelu_alpha, pooling=config['model']['pooling'],
                                  use_bias=use_bias)  
    elif model_name == 'resnet':
        encoder = get_resnet(config=config, model_depth=config['model']['resnet']['model_depth'], channels=channels, lrelu_alpha=lrelu_alpha)
    
    elif model_name == 'densenet':
        encoder = get_desnsenet(config, config['model']['densenet']['model_depth'], channels)

    elif model_name == "ViT":
        ps = config['model']['ViT']['vit_patch_size']
        encoder = ViT(image_size=(depth, height, width), patch_size=(ps,ps,ps),
                    dim=config['model']['ViT']['vit_dim'], depth=config['model']['ViT']['vit_depth'], 
                    heads=config['model']['ViT']['vit_heads'], mlp_dim=config['model']['ViT']['vit_mlp_dim'], pool = 'cls', 
                    channels = channels, dim_head = config['model']['ViT']['vit_dim_head'], emb_dropout = config['model']['ViT']['vit_emb_dropout'],
                    dropout = config['model']['ViT']['vit_dropout_p']) 
        
    elif model_name.lower() == 'transrp':
        if config['model']['TransRP']['image_encoder'] == 'resnet':
            encoder = get_resnet(config=config, model_depth=config['model']['resnet']['model_depth'], channels=channels, lrelu_alpha=lrelu_alpha)
            # these tweaks increase the size of the feature maps for the ViT block (e.g. we remove the downsampling in the last resnet block)
            encoder.backbone.block3[0].conv1.stride = (1, 1, 1)
            encoder.backbone.block3[0].downsample[0].stride = (1,1,1)

        elif config['model']['TransRP']['image_encoder'] == 'densenet':
            encoder = get_desnsenet(config, config['model']['densenet']['model_depth'], channels)
            # these tweaks increase the size of the feature maps for the ViT block (e.g. we remove the downsampling in the last densenet block)
            encoder.features.transition_3.pool.kernel_size = 1
            encoder.features.transition_3.pool.stride = 1
            encoder.features.transition_3.conv.stride = (1,1,1)
        else:
            raise ValueError('Invalid image_encoder for TransRP model: {}.'.format(config['model']['TransRP']['image_encoder']))
    
    else:
        raise ValueError('Invalid model_name: {}.'.format(model_name))
    
    return encoder
