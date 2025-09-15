# -*- coding: utf-8 -*-

from src.models.cnn_pooling import CNN_Pooling
from src.models.resnet import get_resnet
from src.models.densenet import get_densenet, get_short_densenet
from src.models.convnext import get_convnext, get_short_convnext
from src.models.efficientnetv2 import get_efficientnetv2
from src.models.ViT import ViT

def get_encoder(config, channels, depth, height, width):
    """
    Initialize the image encoder.
    Args: 
        config (dict): config params
        channels (int): number of input channels
        depth (int): input depth
        height (int): input height
        width (int): input width
    Returns:
        encoder: the image encoder model
    """

    model_name = config['model']['model_name']

    if model_name == 'cnn_pooling':
        encoder = CNN_Pooling(config=config, n_input_channels=channels)  
    elif model_name == 'resnet':
        encoder = get_resnet(config=config, model_depth=config['model']['resnet']['model_depth'], channels=channels)

    elif ('efficientnetv2' in model_name):
        encoder = get_efficientnetv2(config=config, model_name = model_name, channels=channels)

    elif model_name == 'densenet':
        encoder = get_densenet(config, config['model']['densenet']['model_depth'], channels)

    elif model_name == "convnext":
        encoder = get_convnext(config, config['model']['convnext']['model_size'], channels)
    
    elif model_name == "ViT":
        ps = config['model']['ViT']['vit_patch_size']
        encoder = ViT(image_size=(depth, height, width), patch_size=(ps,ps,ps),
                    dim=config['model']['ViT']['vit_dim'], depth=config['model']['ViT']['vit_depth'], 
                    heads=config['model']['ViT']['vit_heads'], mlp_dim=config['model']['ViT']['vit_mlp_dim'], pool = 'cls', 
                    channels = channels, dim_head = config['model']['ViT']['vit_dim_head'], emb_dropout = config['model']['ViT']['vit_emb_dropout'],
                    dropout = config['model']['ViT']['vit_dropout_p']) 
    
    elif model_name.lower() == 'transrp':
        
        if config['model']['TransRP']['image_encoder'] == 'resnet':
            encoder = get_resnet(config=config, model_depth=config['model']['resnet']['model_depth'], channels=channels)
            # these tweaks increase the size of the feature maps for the ViT block (e.g. we remove the downsampling in the last resnet block)
            
            # delete the last resnet block (now 3 blocks instead of the usual 4)
            # this saves some computation and allows for larger feature maps to the ViT block
            del encoder.backbone.block3

        elif config['model']['TransRP']['image_encoder'] == 'densenet':
            # obtain a densenet model with only 3 blocks (instead of the normal 4)
            encoder = get_short_densenet(config, config['model']['densenet']['model_depth'], channels)
            # these tweaks increase the size of the feature maps for the ViT block (e.g. we remove the downsampling in the last densenet block)

        elif config['model']['TransRP']['image_encoder'] == 'convnext':
            encoder = get_short_convnext(config, config['model']['convnext']['model_size'], channels)
            
        else:
            raise ValueError('Invalid image_encoder for TransRP model: {}.'.format(config['model']['TransRP']['image_encoder']))
    
    else:
        raise ValueError('Invalid model_name: {}.'.format(model_name))
    
    return encoder
