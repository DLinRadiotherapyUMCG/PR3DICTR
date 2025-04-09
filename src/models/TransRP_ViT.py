from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

import monai

#from src.models.linear_layers_OLD import Basic_Output_Head, MultiToxOutputHead
from src.models.ViT import  Transformer
from src.models.tools.get_output_head import get_output_head


class TransRP_ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """
    
    def __init__(
        self,
        config,
        n_features: int,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        clinical_features_method: str = "m3",
        
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            proj_type: position embedding layer type.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.
            clinical_features_method: method to incorporate clinical features in the ViT block.
                "m1": add clinical features as an additional patch to the input of the transformer.
                "m2": append the clinical features to each patch.
                "m3": join the clinical features only in the linear layers at the very end.
                "m4": add clinical features as an additional patch to the input of the transformer, but after the transformer.
        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), proj_type='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), proj_type='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), proj_type='conv', classification=True, spatial_dims=2)

        """


        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        #assert clinical_features_method in ["m1", "m2", "m3", 'mcb'] 

        self.clinical_features_method = clinical_features_method
        self.n_features = n_features
        if self.n_features == 0:
            self.clinical_features_method = "m3"  # no clinical features are incorporated in the ViT block
        
        
        #patch_embedding_hidden_size = hidden_size - n_features if clinical_features_method == "m2" else hidden_size
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size, # new
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        patch_embedding_hidden_size = hidden_size+n_features if clinical_features_method == "m2" else hidden_size

        self.transformer = Transformer(dim=patch_embedding_hidden_size, depth=num_layers, heads=num_heads, dim_head=hidden_size // num_heads, mlp_dim=mlp_dim, dropout=dropout_rate)
        self.norm = nn.LayerNorm(patch_embedding_hidden_size)

        self.img_size  = img_size
        self.patch_num  = int((img_size[0] / patch_size[0] )* (img_size[1] / patch_size[1]) * (img_size[2] / patch_size[2]))

        config["model"]["output_head"]["name"] = "multitox"

        # if self.clinical_features_method == "m3" and self.n_features > 0:   # join the clinical features only in the linear layers at the very end
        #     self.linear_layers = MultiToxOutputHead(config, n_features)
        # else: # clinical features are added to the input of the transformer (m1 or m2), and so we do not need linear layers for this
        #     self.linear_layers = Basic_Output_Head(config)
        
        if self.clinical_features_method == "m3" and self.n_features > 0: 
            n_features_linear_layers = n_features
        else:
            n_features_linear_layers = 0

        
        if self.clinical_features_method ==  "m1": # add clinical features as an additional patch to the input of the transformer
            self.to_label_embedding = nn.Sequential(
                nn.Linear(1, hidden_size), # new
                nn.ReLU(inplace=True)
            )

        elif self.clinical_features_method == "cls":
            #self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            #self.clc_embed = nn.Linear(n_features, hidden_size)
            cls_hidden_dim = config["model"]["TransRP"]["cls_hidden_dim"]
            if config["model"]["TransRP"]["cls_gating"]:
                self.clc_embed = GatedTabularEmbedding(n_features, hidden_size, hidden_dim=cls_hidden_dim)
            else:
                self.clc_embed = TabularEmbedding(n_features, hidden_size, hidden_dim=cls_hidden_dim)

            self.cls_merge_image_patches = config["model"]["TransRP"]["cls_merge_image_features"]
            self.cls_per_class_weighting = config["model"]["TransRP"]["cls_per_class_weighting"]

            if self.cls_merge_image_patches:
                self.alpha = nn.Parameter(torch.tensor(0.5))
            if self.cls_per_class_weighting:
                config["model"]["output_head"]["name"] = "hybridfusion"


        elif self.clinical_features_method == "multi_cls":
            self.N_endpoints = len(config["columns"]["labels"])
            cls_hidden_dim = config["model"]["TransRP"]["cls_hidden_dim"]
            #self.clc_embed_list = [TabularEmbedding(n_features, hidden_size, hidden_dim=cls_hidden_dim) for _ in range(self.N_endpoints)]
            self.clc_embed_list = nn.ModuleList([TabularEmbedding(n_features, hidden_size, hidden_dim=cls_hidden_dim) for _ in range(self.N_endpoints)])
            self.cls_merge_image_patches = False
            self.cls_per_class_weighting = False

            config["model"]["output_head"]["name"] = "multiToken"



        self.linear_layers = get_output_head(config, n_features_linear_layers)



    def forward(self, x, x_clc, vectorize=False):  
        #print("X", x.shape, "x_clc", x_clc.shape)
        x = self.patch_embedding(x)

        if self.clinical_features_method == "m1":      # add clinical features as an additional patch     
            x_clc = x_clc.unsqueeze(-1)

            x_clc = self.to_label_embedding(x_clc)

            x = torch.cat((x, x_clc), dim=1)

        elif self.clinical_features_method == "m2":     # append the clinical features to each patch
            x_clc = x_clc[:, None, :]
            x_clc = x_clc.repeat(1, self.patch_num, 1 )
            
            x = torch.cat((x_clc, x), dim=-1)

        elif self.clinical_features_method == "m2_v2":     # same as m2, but do not make the features vectors longer
            x_clc = x_clc[:, None, :]

            x_clc = x_clc.repeat(1, self.patch_num, 1 )

            x[:, :, :x_clc.shape[-1]] = x_clc

        elif self.clinical_features_method == "cls":     # use the clinical features as the CLS token
            cls_token = self.clc_embed(x_clc).unsqueeze(1)
            #cls_tokens = torch.repeat_interleave(self.cls_token, x.size(0), dim=0)

            x = torch.cat((cls_token, x), dim=1)

        elif self.clinical_features_method == "multi_cls":     # USE MULTIPLE CLS TOKENS
            cls_tokens = [self.clc_embed_list[i](x_clc) for i in range(self.N_endpoints)]
            cls_tokens = torch.stack(cls_tokens, dim=1)

            x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        if self.clinical_features_method == "multi_cls":
            cls_tokens = x[:, :self.N_endpoints, :]  # take the CLS tokens (one per class)
            x = self.linear_layers(cls_tokens, None, vectorize=vectorize)

            return x


        elif "cls" in self.clinical_features_method:  # use the CLS token as the output
            #x = x[:, 0, :]
            if self.cls_per_class_weighting:
                cls_token = x[:, 0, :]
                image_representation = torch.mean(x[:, 1:, :], 1)

                # x = (cls_token, image_representation)

                x = self.linear_layers(cls_token, image_representation, None, vectorize=vectorize)

                return x
            
            elif self.cls_merge_image_patches:  # mix the CLS token with the mean of the image patches
                cls_token = x[:, 0, :]
                image_representation = torch.mean(x[:, 1:, :], 1) # mean of the image patches
                x = self.alpha * cls_token + (1 - self.alpha) * image_representation
            
            else:
                x = x[:, 0, :]
        else:
            x = self.norm(x)                 # use the mean of all of the patches as the output
            x = torch.mean(x, 1)       

        x = self.linear_layers(x, x_clc, vectorize=vectorize)
        
        return x  



class TabularEmbedding(nn.Module):
    def __init__(self, tabular_dim, embed_dim, hidden_dim=128):
        super(TabularEmbedding, self).__init__()
        
        # MLP to project tabular features to the ViT embedding space
        self.fc = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),  # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)    # Output embedding layer
        )
    
    def forward(self, tabular_data):
        return self.fc(tabular_data)  # Projected embedding


class GatedTabularEmbedding(nn.Module):
    def __init__(self, tabular_dim, embed_dim, hidden_dim=128):
        super(GatedTabularEmbedding, self).__init__()
        # Linear layer for projection
        self.fc = nn.Linear(tabular_dim, embed_dim)
        # Gating network to generate a weight vector in [0,1]
        self.gate = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, tabular_data):
        # Compute the base embedding and the gate weights
        embed = self.fc(tabular_data)
        gate_weight = self.gate(tabular_data)
        # Element-wise multiplication to modulate the embedding
        gated_embed = embed * gate_weight
        return gated_embed



def get_transrp_vit(config, n_features : int, feature_map_dim_after_encoder):
    """
    Get the TransRP_ViT model. This function deals with the different image 
    encoders that can be used with the ViT model.
    :param config:
    :param n_features: number of clinical features
    :param feature_map_dim_after_encoder: the dimensions of the feature map after the encoder
    :return: Model
    """
    #image_encoder_name = config['model']['TransRP']['image_encoder'].lower()

    # determine the input size of the ViT block
    in_channels = feature_map_dim_after_encoder[0]
    img_size = feature_map_dim_after_encoder[1:]
    
    ViT_module = TransRP_ViT(
        config=config,
        n_features=n_features,
        in_channels=in_channels,
        img_size=img_size,
        patch_size= config['model']['TransRP']['vit_patch_size'],
        hidden_size = config['model']['TransRP']['vit_dim'],
        mlp_dim = config['model']['TransRP']['vit_mlp_dim'],  
        num_layers = config['model']['TransRP']['vit_depth'],
        num_heads = config['model']['TransRP']['vit_heads'],
        proj_type = 'conv',
        dropout_rate = config['model']['TransRP']['vit_dropout_p'],
        spatial_dims = 3,
        clinical_features_method = config['model']['TransRP']['clinical_features_method'],
    )

    return ViT_module