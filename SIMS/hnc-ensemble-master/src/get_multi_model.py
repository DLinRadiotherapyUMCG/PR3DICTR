# -*- coding: utf-8 -*-

from src.utils.get_encoder import get_encoder
from src.models.linear_layers import MultiToxOutputHead

import torch
from torch import nn

class MultiTox_Classifier(nn.Module):
    def __init__(self, encoder, config, n_features):
        super(MultiTox_Classifier, self).__init__()

        # image encoder backbone (CNN, ResNet, etc.)
        self.encoder = encoder
        self.model_name = config['model']['model_name']
        

        # linear layers
        if "transrp" not in self.model_name:
            self.output_head = MultiToxOutputHead(config=config, n_features=n_features)
            self.flatten = nn.Flatten() # to flatten the output of the encoder
            
    
    def forward(self, x, features, vectorize=False):
        """
        Forward pass of the full model
        """
        # CNN backbone
        x = self.forward_backbone(x, features)

        # MLP
        x_dict = self.forward_linear_layers(x, features, vectorize=vectorize)

        return x_dict
    
    def forward_backbone(self, x, features):  # features are also passed to this function, just for consistency of the forward functions
        """
        Does a forward pass of only the backbone. Extracted features are flattened
        """
        
        if "transrp" in self.model_name: #and "ResNet" in self.model_name:
            if "simple" in self.model_name:
                avg_pool = False
            else:
                avg_pool=True
        else: 
            avg_pool=False
        x = self.encoder(x, autoencoder=avg_pool)
        if self.flatten is not None:
            x = self.flatten(x)

        return x
    
    def forward_linear_layers(self, x, features, vectorize):
        """
        Does a forward pass of only the linear layers (the decision layers)
        Includes the entire output head (clinical, shared and non-shared layers)
        """
        x_dict = self.output_head(x, features, vectorize=vectorize)

        return x_dict

def get_classification_model(config, metadata, save_summary=True):

    
    channels,depth,height,width,n_features = metadata['channels'],metadata['depth'],metadata['height'],metadata['width'],metadata['n_features']
    encoder = get_encoder(config, channels, depth, height, width, n_features)
    # Put the image encoder into a model
    model = MultiTox_Classifier(encoder=encoder, config=config, n_features=n_features)


    # total_params = get_model_summary(config=config, model=model, input_size=[(config.batch_size, channels, depth, height, width), (config.batch_size, max(n_features, 1))], 
    #                                             device=config.device, logger=logger, save_to_file=save_summary)
        
    return model
