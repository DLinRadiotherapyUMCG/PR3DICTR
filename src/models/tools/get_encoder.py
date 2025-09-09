# -*- coding: utf-8 -*-

from src.models.cnn_pooling import CNN_Pooling
from src.models.resnet import get_resnet
from src.models.densenet import get_densenet, get_short_densenet
from src.models.convnext import get_convnext, get_short_convnext
from src.models.modulated_densenet import get_modulated_densenet, get_short_modulated_densenet
from src.models.efficientnetv2 import get_efficientnetv2
from src.models.ViT import ViT
from src.models.HIPT import HIPT

def get_encoder(config, channels, depth, height, width):
    """
    Initialize the image encoder.
    """

    model_name = config['model']['model_name']
    lrelu_alpha = config['model']['lrelu_alpha']

    #dropout_p_endpoint = config.dropout_p_endpoint
    use_bias = config['model']['use_bias']

    # Determine number of downsampling blocks
    # n_down_blocks = [sum([x[0] == 2 for x in strides]), sum([x[1] == 2 for x in strides]),
    #                  sum([x[2] == 2 for x in strides])]

    if model_name == 'cnn_pooling':
        encoder = CNN_Pooling(config=config, n_input_channels=channels, 
                                  lrelu_alpha=lrelu_alpha, use_bias=use_bias)  
    elif model_name == 'resnet':
        encoder = get_resnet(config=config, model_depth=config['model']['resnet']['model_depth'], channels=channels, lrelu_alpha=lrelu_alpha)

    elif ('efficientnetv2' in model_name):
        encoder = get_efficientnetv2(config=config, model_name = model_name, channels=channels)

    elif model_name == 'densenet':
        encoder = get_densenet(config, config['model']['densenet']['model_depth'], channels)

    elif model_name == "modulated_densenet":
        encoder = get_modulated_densenet(config, config['model']['densenet']['model_depth'], channels)

    elif model_name == "convnext":
        encoder = get_convnext(config, config['model']['convnext']['model_size'], channels)
    
    elif model_name == "ViT":
        ps = config['model']['ViT']['vit_patch_size']
        encoder = ViT(image_size=(depth, height, width), patch_size=(ps,ps,ps),
                    dim=config['model']['ViT']['vit_dim'], depth=config['model']['ViT']['vit_depth'], 
                    heads=config['model']['ViT']['vit_heads'], mlp_dim=config['model']['ViT']['vit_mlp_dim'], pool = 'cls', 
                    channels = channels, dim_head = config['model']['ViT']['vit_dim_head'], emb_dropout = config['model']['ViT']['vit_emb_dropout'],
                    dropout = config['model']['ViT']['vit_dropout_p']) 
    
    elif model_name == 'HIPT': # this will never fit on a GPU, don't use it
        encoder = HIPT(channels, depth, height, width)

    elif model_name.lower() == 'transrp':
        

        if config['model']['TransRP']['image_encoder'] == 'resnet':
            encoder = get_resnet(config=config, model_depth=config['model']['resnet']['model_depth'], channels=channels, lrelu_alpha=lrelu_alpha)
            # these tweaks increase the size of the feature maps for the ViT block (e.g. we remove the downsampling in the last resnet block)
            
            # delete the last resnet block (now 3 blocks instead of the usual 4)
            # this saves some computation and allows for larger feature maps to the ViT block
            del encoder.backbone.block3
            # if config['model']['resnet']['model_depth'] < 50:
            #     #encoder.layer4[1].downsample[0].stride = (1, 1, 1)
            #     # encoder.backbone.block3[0].conv1.stride = (1, 1, 1)
            #     # encoder.backbone.block3[0].downsample[0].stride = (1,1,1)
            #     pass
            # else:
            #     encoder.backbone.block3[0].conv2.stride = (1, 1, 1)
            #     encoder.backbone.block3[0].downsample[0].stride = (1,1,1)

        elif config['model']['TransRP']['image_encoder'] == 'densenet':
            # obtain a densenet model with only 3 blocks (instead of the normal 4)
            encoder = get_short_densenet(config, config['model']['densenet']['model_depth'], channels)
            # these tweaks increase the size of the feature maps for the ViT block (e.g. we remove the downsampling in the last densenet block)
            
            # del encoder.features.transition_3

            # encoder.features.transition_3.pool.kernel_size = 1
            # encoder.features.transition_3.pool.stride = 1
            # encoder.features.transition_3.conv.stride = (1,1,1)

        elif config['model']['TransRP']['image_encoder'] == 'modulated_densenet':
            encoder = get_short_modulated_densenet(config, config['model']['densenet']['model_depth'], channels)
            

        elif config['model']['TransRP']['image_encoder'] == 'convnext':
            encoder = get_short_convnext(config, config['model']['convnext']['model_size'], channels)
            
        else:
            raise ValueError('Invalid image_encoder for TransRP model: {}.'.format(config['model']['TransRP']['image_encoder']))
    
    else:
        raise ValueError('Invalid model_name: {}.'.format(model_name))
    
    return encoder
