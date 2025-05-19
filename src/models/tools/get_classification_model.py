# -*- coding: utf-8 -*-
import os
import torch
from torch import nn

from src.constants import DEVICE
from src.models.tools.get_encoder import get_encoder
#from src.models.temp_ff_linear_layers import MultiToxOutputHead   # TODO: ask Luuk whats going on here
from src.models.tools.model_summary import get_model_summary
from src.models.tools.get_output_head import get_output_head
from src.models.TransRP_ViT import get_transrp_vit


class MultiTox_Classifier(nn.Module):
    def __init__(self, encoder, config, n_features : int, metadata=None):
        super(MultiTox_Classifier, self).__init__()

        # image encoder backbone (CNN, ResNet, etc.)
        self.encoder = encoder
        self.model_name = config['model']['model_name'].lower()
        self.modulated_backbone = config['model']['modulated_backbone']

        if self.modulated_backbone:
            self.n_modulated_features = config['model']['n_clinical_features_in_backbone']
            n_features = n_features - self.n_modulated_features # add 1 for the modulated proton feature
        self.n_features = n_features
        
        # get the output head module (e.g. linear layers)
        if "transrp" in self.model_name:
            self.flatten = None

            feature_map_dim_after_encoder = self.determine_image_encoder_output_dim(metadata) # find the feature map dimensions of the image encoder's output
            self.output_head = get_transrp_vit(config, n_features, feature_map_dim_after_encoder)
            
        else:
            self.flatten = nn.Flatten()

            self.output_head = get_output_head(config, n_features)

            
            
        
    
    def forward(self, x, features=None, vectorize=False):
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

        if self.modulated_backbone:
            #x = self.encoder(x, features=features[:, :])
            x = self.encoder(x, features=features[:, :self.n_modulated_features], autoencoder=avg_pool)
        else:
            x = self.encoder(x, autoencoder=avg_pool)
        
        if self.flatten is not None:
            x = self.flatten(x)

        return x
    
    def forward_output_head(self, x, features, vectorize):
        """
        Does a forward pass of only the linear layers (the decision layers)
        Includes the entire output head (clinical, shared and non-shared layers)
        """
        if self.modulated_backbone:
            #x = self.encoder(x, features=features[:, :])
            clc_features = features[:, self.n_modulated_features:]
        else:
            clc_features = features

        #x_dict = self.output_head(x, features[:, :], vectorize=vectorize)

        x_dict = self.output_head(x, clc_features, vectorize=vectorize)

        return x_dict
    

    def determine_image_encoder_output_dim(self, metadata):
        """
        The TRP model requires the output dimensions of the image encoder, to define the ViT module. 
        This function determines the output dimensions of the image encoder by doing one forward pass of it.
        """
        self.eval()
        with torch.no_grad():
            temp_x_input = torch.zeros((1, metadata['channels'], metadata['depth'], metadata['height'], metadata['width'])).to(DEVICE)
            temp_clc_features = torch.zeros((1, self.n_features)).to(DEVICE) if self.modulated_backbone else None
            #print(DEVICE)
            self.encoder.to(DEVICE)
            x = self.forward_image_encoder(temp_x_input, temp_clc_features)
            #x = self.encoder(x)
            return x.shape[1:]



def get_classification_model(config, metadata, save_summary=True):
  
    channels,depth,height,width,n_features = metadata['channels'], metadata['depth'], metadata['height'], metadata['width'], metadata['n_features']
    encoder = get_encoder(config, channels, depth, height, width)
    # Put the image encoder into a model
    model = MultiTox_Classifier(encoder=encoder, config=config, n_features=n_features, metadata=metadata)


    get_model_summary(config=config, model=model, input_size=[(config['training']['batch_size'], channels, depth, height, width), (config['training']['batch_size'], max(n_features, 1))], 
                                                device=DEVICE, save_to_file=save_summary)
        
    return model

