# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import logging

from src.constants import DEVICE
from src.models.tools.get_encoder import get_encoder
from src.models.tools.model_summary import get_model_summary
from src.models.tools.get_output_head import get_output_head
from src.models.TransRP_ViT import get_transrp_vit
from src.models.MLP import MultiLayerPerceptron

class ImageClassifier(nn.Module):
    def __init__(self, encoder, config, n_features : int, metadata = None):
        super(ImageClassifier, self).__init__()

        # image encoder backbone (CNN, ResNet, etc.)
        self.encoder = encoder
        self.model_name = config['model']['model_name'].lower()
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

        x = self.encoder(x, autoencoder=avg_pool)
        
        if self.flatten is not None:
            x = self.flatten(x)

        return x
    
    def forward_output_head(self, x, features, vectorize):
        """
        Does a forward pass of only the linear layers (the decision layers)
        Includes the entire output head (clinical, shared and non-shared layers)
        """
        clc_features = features

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
            temp_clc_features = None 
            
            self.encoder.to(DEVICE)
            x = self.forward_image_encoder(temp_x_input, temp_clc_features)
            print(x.shape)
            return x.shape[1:]





def get_classification_model(config, metadata, save_summary=True):
  
    channels,depth,height,width,n_features = metadata['channels'], metadata['depth'], metadata['height'], metadata['width'], metadata['n_features']


    if channels > 0: # if images are present
        logging.info("Creating image model")
        
        encoder = get_encoder(config, channels, depth, height, width)
        # Put the image encoder into a model
        model = ImageClassifier(encoder=encoder, config=config, n_features=n_features, metadata=metadata)

    else:
        logging.info("No image_keys present. Creating clinical-only model (MLP)")
        channels, depth, height, width = 1, 1, 1, 1 # dummy values for model summary
        model = MultiLayerPerceptron(encoder=None, config=config, n_features=n_features, metadata=metadata)

    get_model_summary(config=config, model=model, input_size=[(config['training']['batch_size'], channels, depth, height, width), (config['training']['batch_size'], max(n_features, 1))], 
                                                device=DEVICE, save_to_file=save_summary)
        
    return model

