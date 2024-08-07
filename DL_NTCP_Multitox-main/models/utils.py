import torch
import os
import math

from torchinfo import summary

from models.resnet_dcnn_lrelu import ResNet_DCNN_LReLU
from models.dcnn_lrelu import DCNN_LReLU
from models.dcnn_pooling import DCNN_Pooling
from models.resnet_lrelu import get_resnet_lrelu, get_resnet
from models.resnext_lrelu import get_resnext_relu
from models.efficientnetv2 import get_efficientnetv2
from models.ViT import ViT
from models.MedViT import MedViT_small, MedViT_base, MedViT_large
from models.swin_3D import Swin_tiny, Swin_small, Swin_base, Swin_large
from models.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from models.TransRP import TransRP_ResNet18, TransRP_DenseNet121
from models.SimpleTransRP import SimpleTransRP
from models.densenet import get_desnsenet

# the main model class
from models.MT_model import MultiTox_Autoencoder, MultiTox_Classifier, MultiTox_MLP



def get_encoder(config, channels, depth, height, width, n_features):
    model_name = config.model_name
    filters = config.filters
    kernel_sizes = config.kernel_sizes
    strides = config.strides
    pad_value = config.pad_value
    lrelu_alpha = config.lrelu_alpha
    pooling_conv_filters = config.pooling_conv_filters
    perform_pooling = config.perform_pooling
    endpoint_list = config.endpoint_list

    #dropout_p_endpoint = config.dropout_p_endpoint
    use_bias = config.use_bias

    if config.siamese_encoder:
        channels = 1

    """
    Initialize model.
    """
    # Determine number of downsampling blocks
    n_down_blocks = [sum([x[0] == 2 for x in strides]), sum([x[1] == 2 for x in strides]),
                     sum([x[2] == 2 for x in strides])]

    if model_name == 'resnet_dcnn_lrelu':
        encoder = ResNet_DCNN_LReLU(config=config,
                                  n_input_channels=channels, depth=depth, height=height, width=width,
                                  filters=filters, kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                  n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha,
                                  pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                  use_bias=use_bias)
        
    # linear_units, dropout_p, linear_units_endpoint, num_ohe_classes
                 # endpoint_list, n_features, clinical_variables_position, clinical_variables_linear_units

    elif model_name == 'dcnn_lrelu':
        encoder = DCNN_LReLU(config=config,
                                n_input_channels=channels, depth=depth, height=height, width=width,
                                  filters=filters, kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                  n_down_blocks=n_down_blocks, lrelu_alpha=lrelu_alpha,
                                  pooling_conv_filters=pooling_conv_filters, perform_pooling=perform_pooling,
                                  n_features=n_features, use_bias=use_bias)
    elif model_name == 'dcnn_pooling':
        encoder = DCNN_Pooling(config=config,
                                n_input_channels=channels,n_features=n_features, 
                                  filters=filters, kernel_sizes=kernel_sizes, strides=strides, pad_value=pad_value,
                                  lrelu_alpha=lrelu_alpha, pooling=config.pooling,
                                  use_bias=use_bias)  
    elif model_name == 'resnet_lrelu':
        encoder = get_resnet_lrelu(config, model_depth=config.resnet_model_depth, channels=channels, n_features=n_features, 
                                 lrelu_alpha=lrelu_alpha, filters=filters)
    elif model_name == "resnext_lrelu":
        encoder = get_resnext_relu(config, model_depth=config.resnet_model_depth, channels=channels, n_features=n_features)
    elif "efficientnetv2" in model_name:
        encoder = get_efficientnetv2(config, model_name="efficientnetv2_l", channels=channels, n_features=n_features)
    elif model_name == "ViT":
        ps = config.vit_patch_size
        encoder = ViT(config=config, n_features=n_features, image_size=(depth, height, width), patch_size=(ps,ps,ps),
                    dim=config.vit_dim, depth=config.vit_depth, heads=config.vit_heads, mlp_dim=config.vit_mlp_dim, pool = 'cls', 
                    channels = channels, dim_head = config.vit_dim_head, emb_dropout = 0.) 
    elif model_name == "MedViT_small":
        encoder = MedViT_small(config, n_features, channels)
    elif model_name == "MedViT_base":
        encoder = MedViT_base(config, n_features, channels)
    elif model_name == "MedViT_large":
        encoder = MedViT_large(config, n_features, channels)
    
    elif 'swin3d' in model_name.lower():
        if "tiny" in model_name.lower():
            encoder = Swin_tiny(config, n_features, channels)
        elif "small" in model_name.lower():
            encoder = Swin_small(config, n_features, channels)
        elif "base" in model_name.lower():
            encoder = Swin_base(config, n_features, channels)
        elif "large" in model_name.lower():
            encoder = Swin_large(config, n_features, channels)

    # Swin_tiny, , Swin_base, 
    # elif model_name == "Swin3D":
    #     model = SwinTransformer3D(config, n_features, patch_size=config.swin_patch_size, window_size=config.swin_window_size, 
    #                               depths=config.swin_depths, num_heads=config.swin_num_heads, embed_dim=config.swin_embed_dim)
    elif "transrp" in model_name.lower():
        #print(model_name)
        if "resnet" in model_name.lower():
            encoder = get_resnet(config, model_depth=config.resnet_model_depth, channels=channels, n_features=n_features, 
                                    lrelu_alpha=lrelu_alpha, filters=filters)
            # set outout feature map with size of 6x6x6x512
            if "simpletransrp" not in model_name.lower():
                encoder.backbone.block3[0].conv1.stride = (1, 1, 1)
                encoder.backbone.block3[0].downsample[0].stride = (1, 1, 1)
        elif "densenet" in model_name.lower():
            encoder = get_desnsenet(config, model_depth=config.densenet_model_depth, channels=channels)

            encoder.backbone.transition_3.pool.kernel_size = 1
            encoder.backbone.transition_3.pool.stride = 1
            encoder.backbone.transition_3.conv.stride = (1,1,1)

        elif "resnext" in model_name.lower():
            encoder = get_resnext_relu(config, model_depth=config.resnet_model_depth, channels=channels, n_features=n_features)
            #print(encoder)
            if config.resnet_model_depth <= 34: # Basic blocks
                encoder.backbone.block3[0].conv1.stride = (1, 1, 1)
            else: # Bottleneck blocks
                encoder.backbone.block3[0].conv2.stride = (1, 1, 1)
            encoder.backbone.block3[0].downsample[0].stride = (1, 1, 1)

        elif "convnext" in model_name.lower():
            patch_size = 4
            kernel_size = 5
            drop_first_block = False
            encoder = convnext_small(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)

            #encoder.downsample_layers[3] = encoder.downsample_layers[3][0]
            encoder.downsample_layers[3][1].kernel_size = (1,1,1)
            encoder.downsample_layers[3][1].stride = (1,1,1)
            encoder.downsample_layers[3][1].padding = "same"

            #print(encoder)

            # if 'xlarge' in model_name.lower():
            #     encoder = convnext_xlarge(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
            # elif 'tiny' in model_name.lower():
            #     encoder = convnext_tiny(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
            # elif 'small' in model_name.lower():
            #     encoder = convnext_small(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
            # elif 'base' in model_name.lower():
            #     encoder = convnext_base(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
            # elif 'large' in model_name.lower():
            #     encoder = convnext_large(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)



    elif "convnext" in model_name.lower():
        patch_size = config.convnext_patch_size
        kernel_size = config.convnext_kernel_size
        drop_first_block = config.convnext_drop_first_block
        if 'xlarge' in model_name.lower():
            encoder = convnext_xlarge(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
        elif 'tiny' in model_name.lower():
            encoder = convnext_tiny(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
        elif 'small' in model_name.lower():
            encoder = convnext_small(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
        elif 'base' in model_name.lower():
            encoder = convnext_base(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
        elif 'large' in model_name.lower():
            encoder = convnext_large(config, n_features, channels, kernel_size=kernel_size, patch_size=patch_size, drop_first_block=drop_first_block)
    # elif model_name == "TransRP_ResNet18":
    #     encoder = TransRP_ResNet18(in_channels=512, in_channels_cnn=3, img_size=[6,6,6], patch_size=[1,1,1])
    # elif model_name == "TransRP_DenseNet121":
    #     encoder = TransRP_DenseNet121(in_channels=1024, in_channels_cnn=3, img_size=[6,6,6], patch_size=[1,1,1])
    
            
    elif "densenet" in model_name.lower():
        encoder = get_desnsenet(config, model_depth=config.densenet_model_depth, channels=channels)
        # from torch import nn
        # import monai
        # encoder = nn.Sequential( *list((monai.networks.nets.resnet18(spatial_dims=3, n_input_channels = channels, num_classes=  1, conv1_t_stride = 2)).children())[0:8])
        
        # (encoder)[7][0].conv1.stride = (1, 1, 1) # set outout feature map with size of 6x6x6x512
        # (encoder)[7][0].downsample[0].stride = (1, 1, 1)
        #encoder = TransRP_DenseNet121(in_channels=1024, in_channels_cnn=3, img_size=[6,6,6], patch_size=[1,1,1])


    else:
        raise ValueError('Invalid model_name: {}.'.format(model_name))

    if config.siamese_encoder:
        from models.MT_model import SiameseEncoder
        encoder = SiameseEncoder(encoder)
    
    return encoder


def get_classification_model(config, channels, depth, height, width, n_features, pretrained_path, logger, save_summary=True):

    model_name = config.model_name

    if model_name == 'MLP':
        model = MultiTox_MLP(config, n_features)
    # elif "simpletransrp" in model_name.lower():
    #     encoder = get_encoder(config, channels, depth, height, width, n_features)
    #     model = SimpleTransRP(encoder=encoder, config=config, n_features=n_features)
    else:
        encoder = get_encoder(config, channels, depth, height, width, n_features)
        # Put the image encoder into a model
        model = MultiTox_Classifier(encoder=encoder, config=config, n_features=n_features)

        #print(model)


    total_params = get_model_summary(config=config, model=model, input_size=[(config.batch_size, channels, depth, height, width), (config.batch_size, max(n_features, 1))], 
                                                device=config.device, logger=logger, save_to_file=save_summary)
    

    # Load pretrained model weights
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=config.device)

        if model_name == 'resnet_original':
            # The pretrained model assumes 1 input channel, i.e. input_shape = (B, 1, D, H, W). However
            # we use input_shape = (B, 3, D, H, W). Therefore we repeat the filter weights across the different channels.
            # This is only required for the very first layer.
            module_conv1_weight = checkpoint['state_dict']['module.conv1.weight']
            checkpoint['state_dict']['module.conv1.weight'] = module_conv1_weight.repeat(1, channels, 1, 1, 1)

            # Only consider layers that are present in the newly initialized model and in the pretrained model
            model_dict = model.state_dict()
            # Somehow the pretrained model has 'module.' preceding the layer's name
            pretrain_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if
                             k.replace('module.', '') in model_dict.keys()}
            if len(pretrain_dict) == 0:
                logger.my_print('checkpoint["state_dict"].keys():', checkpoint['state_dict'].keys())
                logger.my_print('model_dict.keys():', model_dict.keys())
                raise ValueError('Invalid pretrain_dict. No weights gets transferred.\nPretrain_dict: {}.'.
                                 format(pretrain_dict))

            # update(): updates key-value pairs in model_dict with key-value pairs in pretrain_dict
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)

            # Check
            # TODO: quick check (remove later)
            pname, p = next(model.named_parameters())
            # pname: parameter name
            # p: parameter weights
            logger.my_print('pname: {}.'.format(pname))
            logger.my_print(p.shape)
            # Check for elementwise-equality
            check_equality = torch.eq(p.data.cpu(), checkpoint['state_dict']['module.conv1.weight'].cpu())
            if False in check_equality:
                raise ValueError('Check of equality failed.')
                assert False
        else:
            if config.transfer_learning_mode:
                pretrain_dict = torch.load(pretrained_path, map_location=config.device)
                model_dict = model.state_dict()
                # pretrain dict has only the encoder weights !!
                pretrain_dict = {k: v for k, v in pretrain_dict.items() if ((k in model_dict.keys()) and ('output_head' not in k and 'decoder' not in k))}
                model_dict.update(pretrain_dict)
                
                
                #model.load_state_dict(model_dict)

                # TRANSFER LEARNING: load the weights of the encoder only
                keys = []
                with torch.no_grad():
                    for key, param_target in model.named_parameters():
                        
                        if key in pretrain_dict.keys():
                            #print(key)
                            keys.append(key)
    
                            param_source = pretrain_dict[key]
                            #print(param_target.shape, param_source.shape)

                            if model_name == 'ViT' and 'to_patch_embedding' in key:
                                if param_target.shape != param_source.shape:  # if the channels do not match, then there will a mismatch in the ViT embedding size
                                    #print(param_target.shape, param_source.shape)
                                    n = param_target.shape[-1] // param_source.shape[-1]

                                    if len(param_source.shape) == 1:
                                        #param_source = param_source.unsqueeze(0)
                                        param_source = param_source.repeat(n)     # just stack them until it fits again
                                    elif len(param_source.shape) == 2:
                                        param_source = param_source.repeat(1, n)
                                    

                            
                            param_target.data.copy_(param_source.data)
                logger.my_print('Loaded weights for {} keys.'.format(len(keys)))

                # make a new output head
                # from models.linear_layers import MultiToxOutputHead
                # new_output_head = MultiToxOutputHead(config=config, n_features=n_features)
                # model.output_head = new_output_head
            else:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except:
                    model.load_state_dict(checkpoint)

    
    # Initialize the weights of the model (if not loading a pretrained model)
    if "medvit" not in model_name.lower():
        model_weights_settings(config=config, model=model, pretrained_path_i=pretrained_path, logger=logger)
        
    return model, total_params


def get_autoencoder_model(config, channels, depth, height, width, n_features, logger, save_summary=True):
    """
    Makes an autoecoder model
    """
    encoder = get_encoder(config, channels, depth, height, width, n_features)
    
    model = MultiTox_Autoencoder(encoder=encoder, input_shape=(config.batch_size, channels, depth, height, width))

    model.to(config.device)
    model.build_decoder(input_shape=(config.batch_size, channels, depth, height, width), device=config.device)

    total_params = get_model_summary(config=config, model=model, input_size=[(config.batch_size, channels, depth, height, width)], 
                                                device=config.device, logger=logger, save_to_file=save_summary)

    return model


def model_weights_settings(config, model, pretrained_path_i, logger):
    # Initialize the weights of the model

    if 'selu' in config.model_name:
        # Source: https://github.com/bioinf-jku/SNNs/tree/master/Pytorch
        config.weight_init_name = 'kaiming_normal'
        config.kaiming_a = None
        config.kaiming_mode = 'fan_in'
        config.kaiming_nonlinearity = 'linear'

    logger.my_print('Weight init name: {}.'.format(config.weight_init_name))
    if pretrained_path_i is not None:
        logger.my_print('Using pretrained weights: {}.'.format(pretrained_path_i))
    elif ((pretrained_path_i is None) and
            (config.weight_init_name is not None) and
            ('efficientnet' not in config.model_name) and
            ('convnext' not in config.model_name.lower()) and
            ('resnext' not in config.model_name)):
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        model.apply(lambda m: weights_init(m=m, weight_init_name=config.weight_init_name, kaiming_a=config.kaiming_a,   # TODO: move these variables into misc.py
                                                kaiming_mode=config.kaiming_mode,
                                                kaiming_nonlinearity=config.kaiming_nonlinearity, gain=config.gain,
                                                logger=logger))
        logger.my_print('Using our own weights init scheme for {}.'.format(config.model_name))
    else:
        logger.my_print('Using default PyTorch weights init scheme for {}.'.format(config.model_name))
    
    return model




def get_model_summary(config, model, input_size, device, logger, save_to_file = True):
    
    """
    Get model summary and number of trainable parameters.

    Args:
        model:
        input_size:
        device:
        logger:

    Returns:
        total_params (int): number of trainable parameters

    """
    # Get and save summary
    verbose = 0 if config.hyperparameter_tuning_mode else 1
    txt = str(summary(model=model, input_size=input_size, device=device, verbose=verbose, depth=5,
                      col_names=["input_size","output_size","num_params", "kernel_size"], row_settings=["var_names"]))
    if save_to_file:
        file = open(os.path.join(config.exp_dir, config.filename_model_txt), 'a+', encoding='utf-8')
        file.write(txt)
        file.close()

    # Determine number of trainable parameters
    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    # total_params = sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params








def weights_init(m, weight_init_name, kaiming_a, kaiming_mode, kaiming_nonlinearity, gain, logger):
    """
    Custom weights initialization.

    The weights_init function takes an initialized model as input and re-initializes all convolutional,
    batch normalization and instance normalization layers.

    Source: https://pytorch.org/docs/stable/nn.init.html
    Source: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/conv.py#L49
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/linear.py#L83
    Source: https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/batchnorm.py#L51
    Source: https://discuss.pytorch.org/t/initialization-and-batch-normalization/30655/2

    Args:
        m:
        weight_init_name:
        kaiming_a:
        kaiming_mode:
        kaiming_nonlinearity:
        gain:
        logger:

    Returns:

    """
    # Initialize variables
    classname = m.__class__.__name__

    if ('Conv' in classname) or ('Linear' in classname):
        if weight_init_name == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'uniform':
            torch.nn.init.uniform_(m.weight)
        elif weight_init_name == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        elif weight_init_name == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(m.weight, a=kaiming_a, mode=kaiming_mode, nonlinearity=kaiming_nonlinearity)
        elif weight_init_name == 'normal':
            torch.nn.init.normal_(m.weight)
        elif weight_init_name == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
        elif weight_init_name == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight, gain=gain)
        else:
            raise ValueError('Invalid weight_init_name: {}.'.format(weight_init_name))

        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)

