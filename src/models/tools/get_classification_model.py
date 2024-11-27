# -*- coding: utf-8 -*-
import os
from src.utils.get_encoder import get_encoder
#from src.models.temp_ff_linear_layers import MultiToxOutputHead   # TODO: ask Luuk whats going on here
from src.models.linear_layers import MultiToxOutputHead


import torch
from torch import nn
from src.utils.fileHandler import create_file, create_folder
from src.models.tools.model_summary import get_model_summary
from src.models.TransRP_ViT import get_transrp_vit


class MultiTox_Classifier(nn.Module):
    def __init__(self, encoder, config, n_features : int, metadata=None):
        super(MultiTox_Classifier, self).__init__()

        # image encoder backbone (CNN, ResNet, etc.)
        self.encoder = encoder
        self.model_name = config['model']['model_name'].lower()
        
        # linear layers
        if "transrp" not in self.model_name:
            self.output_head = MultiToxOutputHead(config=config, n_features=n_features)
            self.flatten = nn.Flatten() # to flatten the output of the encoder
        else:
            self.flatten = None
            feature_map_dim_after_encoder = self.determine_image_encoder_output_dim(metadata) # find the feature map dimensions of the image encoder's output
            self.output_head = get_transrp_vit(config, n_features, feature_map_dim_after_encoder) # define the ViT module
            
            
    
    def forward(self, x, features, vectorize=False):
        """
        Forward pass of the full model
        """
        # CNN backbone
        x = self.forward_image_encoder(x, features)

        # MLP
        x_dict = self.forward_output_head(x, features, vectorize=vectorize)

        return x_dict
    
    def forward_image_encoder(self, x, features):  # features are also passed to this function, just for consistency of the forward functions
        """
        Does a forward pass of only the image encoder. Extracted features are flattened
        """
        
        if "transrp" in self.model_name: #and "ResNet" in self.model_name:
            avg_pool=True
        else: 
            avg_pool=False
        x = self.encoder(x, autoencoder=avg_pool)
        if self.flatten is not None:
            x = self.flatten(x)

        return x
    
    def forward_output_head(self, x, features, vectorize):
        """
        Does a forward pass of only the linear layers (the decision layers)
        Includes the entire output head (clinical, shared and non-shared layers)
        """
        x_dict = self.output_head(x, features, vectorize=vectorize)

        return x_dict
    

    def determine_image_encoder_output_dim(self, metadata):
        """
        The TRP model requires the output dimensions of the image encoder, to define the ViT module. 
        This function determines the output dimensions of the image encoder by doing one forward pass of it.
        """
        self.eval()
        with torch.no_grad():
            x = self.forward_image_encoder(torch.zeros((1, metadata['channels'], metadata['depth'], metadata['height'], metadata['width'])), None)
            #x = self.encoder(x)
            return x.shape[1:]




def get_classification_model(config, metadata, save_summary=True):
  
    channels,depth,height,width,n_features = metadata['channels'],metadata['depth'],metadata['height'],metadata['width'],metadata['n_features']
    encoder = get_encoder(config, channels, depth, height, width, n_features)
    # Put the image encoder into a model
    model = MultiTox_Classifier(encoder=encoder, config=config, n_features=n_features, metadata=metadata)

    get_model_summary(config=config, model=model, input_size=[(config['training']['batch_size'], channels, depth, height, width), (config['training']['batch_size'], max(n_features, 1))], 
                                                device=config['general']['device'], save_to_file=save_summary)
        
    return model

