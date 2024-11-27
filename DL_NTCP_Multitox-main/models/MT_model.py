import torch
from torch import nn
from models.linear_layers import MultiToxOutputHead, Basic_Output_Head
from models.TransRP_ViT import TransRP_ViT
from models.SimpleTransRP import SimpleTransRP
from models.TransRP_DeiT import TransRP_DieT


"""
MultiToxicity classifier model
consists of an image encoder backbone, and the output head 
(linear layers) which also takes in clinical variables
"""
class MultiTox_Classifier(nn.Module):
    def __init__(self, encoder, config, n_features):
        super(MultiTox_Classifier, self).__init__()

        # image encoder backbone (CNN, ResNet, etc.)
        self.encoder = encoder
        self.model_name = config.model_name.lower()
        

        # linear layers
        if "transrp" not in self.model_name:
            self.output_head = MultiToxOutputHead(config=config, n_features=n_features)
            self.flatten = nn.Flatten() # to flatten the output of the encoder

        else:
            self.flatten = None
            clinical_features_method = config.model_name.split("_")[-1]
            if "resnet" in self.model_name:
                
                #in_channels=512
                if config.resnet_model_depth >= 50: 
                    in_channels=2048
                elif config.resnet_model_depth == 10:
                    in_channels = 256
                else:
                    in_channels=512
                img_size=[6,6,6]
                patch_size=[1,1,1]
            elif "resnext" in self.model_name:
                #in_channels=2048
                if config.resnet_model_depth <= 34:
                    in_channels = 256
                else: # config.resnet_model_depth >= 50:
                    in_channels = 1024 
                img_size=[6,6,6]
                patch_size=[1,1,1]
            elif "densenet" in self.model_name:                
                #densenet_model_depth = 169  # 121 (default), 169, 201, 264
                if config.densenet_model_depth == 121:
                    in_channels=1024
                elif config.densenet_model_depth == 169:
                    in_channels = 1664
                elif config.densenet_model_depth == 201:
                    in_channels = 1920
                else: # depth = 264
                    in_channels = 2688
                img_size=[6,6,6]
                patch_size=[1,1,1]
            elif "convnext" in self.model_name:
                in_channels=768
                img_size=[6,6,6]
                patch_size=[1,1,1]


                #print("in_channels", in_channels)
            if "heads" in self.model_name:
                num_layers= 4
                num_heads = 24
            else:
                num_layers= 12
                num_heads= 12
                
            
            self.output_head = TransRP_ViT(config, n_features, in_channels=in_channels, img_size=img_size, patch_size=patch_size, clinical_features_method=clinical_features_method,
                                           dropout_rate=config.dropout_p, num_layers=num_layers, num_heads=num_heads)
            # self.output_head = TransRP_DieT(config, n_features, in_channels=in_channels, img_size=img_size, patch_size=patch_size, clinical_features_method=clinical_features_method,
            #                                dropout_rate=config.dropout_p, num_layers=num_layers, num_heads=num_heads)
            
            if 'simple' in self.model_name:
                num_heads = 32
                num_heads = 64
                self.flatten = nn.Flatten()
                #self.attention_layers = 
                self.output_head = SimpleTransRP(config, input_size=in_channels, n_features=n_features, num_heads=num_heads)
    # def forward(self, x, features):
    #     x = self.encoder(x)
    #     x = self.flatten(x)
    #     x_dict = self.output_head(x, features)

    #     return x_dict
    
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
    



class MultiTox_MLP(nn.Module):
    """
    A simple MLP model for the MultiTox dataset. 
    Ignores the clinical backbone, and instead feeds the clinical variables through the shared layers
    """
    def __init__(self, config, n_features):
        super(MultiTox_MLP, self).__init__()

        # image encoder backbone (CNN, ResNet, etc.)
        self.device = config.device
        self.encoder = None
        self.model_name = config.model_name
        self.output_head = MultiToxOutputHead(config=config, n_features=0)    # NOTE: set n_features=0 to ensure that no clinical layers are made
  
    
    def forward(self, x, features, vectorize=False):
        """
        Forward pass of the full model
        """
        # CNN backbone
        #x = self.forward_backbone(x, features)

        # MLP
        x_dict = self.forward_linear_layers(x=features, features=None, vectorize=vectorize)   # insert the clinical features as the X input, so that it goes through the shared layers

        return x_dict
    
    
    def forward_linear_layers(self, x, features, vectorize):
        """
        Does a forward pass of only the linear layers (the decision layers)
        Includes the entire output head (clinical, shared and non-shared layers)
        """
        x_dict = self.output_head(x, features, vectorize=vectorize)

        return x_dict









class SiameseEncoder(nn.Module):
    def __init__(self, encoder):
        super(SiameseEncoder, self).__init__()

        self.encoder = encoder

    def forward(self, x):
        encoded_outputs = []
        #print(x.)
        # one forward pass per image channel
        for i in range(x.size(1)):
            encoded_outputs.append(self.encoder(x[:,i].clone().unsqueeze(1)))
        #x = self.encoder(x.clone())
        #print(encoded_outputs[0].shape, encoded_outputs[1].shape, encoded_outputs[2].shape)

        # stack the encoded outputs together
        encoded_outputs = torch.concat(encoded_outputs, dim=1)
        #print(encoded_outputs.shape)

        return encoded_outputs


"""
MT Autoencoder model, used mostly to train an encoder
"""
class MultiTox_Autoencoder(nn.Module):
    def __init__(self, encoder, input_shape):
        super(MultiTox_Autoencoder, self).__init__()

        self.encoder = encoder

        
    def build_decoder(self, input_shape, device):
        with torch.no_grad():
            dummy_input = torch.randn(*input_shape, device=device)
            encoder_output_shape = self.encoder(dummy_input, autoencoder=True).shape
        
        self.decoder = AdaptiveDecoder(encoder_output_shape, input_shape)

    def forward(self, x, features=None):
        x = self.encoder(x, autoencoder=True)   # NOTE: autoencoder=True enforces that no average pooling is done at the end of the encoder
        x = self.decoder(x)

        return x
    
    def forward_encoder(self, x):
        return self.encoder(x, autoencoder=True)
    

"""
An adaptive decoder that takes in the size of the encoder's output, 
and expands it until it is the same size as the original input
"""
class AdaptiveDecoder(nn.Module):
    def __init__(self, encoder_output_shape, input_shape):
        super(AdaptiveDecoder, self).__init__()
        
        self.decoder = self._build_decoder(encoder_output_shape, input_shape)

    def _build_decoder(self, encoder_output_shape, input_shape):
        layers = []
        current_shape = encoder_output_shape
        
        while current_shape[2] < input_shape[2]:
            out_channels = current_shape[1] // 2
            layers.append(nn.ConvTranspose3d(current_shape[1], out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
            current_shape = (current_shape[0], out_channels, current_shape[2]*2, current_shape[3]*2, current_shape[4]*2)
        
        layers.append(nn.ConvTranspose3d(current_shape[1], 1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)