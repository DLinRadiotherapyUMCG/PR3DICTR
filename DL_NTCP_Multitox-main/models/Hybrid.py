import torch
from torch import nn
from models.linear_layers import MultiToxOutputHead
from models.TransRP_ViT import TransRP_ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from models.ViT import Transformer
from models.linear_layers import Basic_Output_Head, MultiToxOutputHead


class CrossAttention(nn.Module):
    def __init__(self, attention_dim, query_input_dim, num_heads, dropout=0.1): # input 1 is K and V, 2 is Q
        super().__init__()

        self.projection_query = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(query_input_dim, attention_dim),
            nn.ReLU()
        )
        self.projection_query = nn.Linear(query_input_dim, attention_dim)
        

        #print(dim_input_1, num_heads)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, input_1, input_2):
        Q = self.projection_query(input_2)

        K = input_1
        V = input_1

        attention_output, _ = self.attention(Q, K, value=V)

        return attention_output


class SimpleTransRP(nn.Module):                  # NOTE: EMBED SIDE DOES NOT GET USED !
    def __init__(self, config, input_size = 256, n_features = 0, num_heads = 1):
        super(SimpleTransRP, self).__init__()

        self.skip_connection = config.skip_connection

        self.mlp_dim = 512

        self.self_attention = SelfAttentionBlock(input_size, num_heads=32, hidden_size=256, dropout_rate=config.dropout_p)


        self.clinical_features_encoder = nn.Sequential(  # linear layers to increase the dimension of the clinical features
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(256)
        )
        # input 1 is K and V, imput 2 is Q
        self.cross_attention_images = CrossAttention(input_size, 256, num_heads, dropout=config.dropout_p)
        self.cross_attention_features = CrossAttention(256, input_size, num_heads, dropout=config.dropout_p)


        self.linear_layers = Basic_Output_Head(config)


    def forward(self, x, features, vectorize=False):
        """
        
        """

        
        features = self.clinical_features_encoder(features)
       # print(x.shape, features.shape)
        #print("DIM X ", x.shape, "DIM F ", features.shape)
        i_attention_block = self.cross_attention_images(x, features)
        #print("DIM I_ATTENTION ", i_attention_block.shape, "DIM X ", x.shape)
        i_attention_block += x

        
        f_attention_block = self.cross_attention_features(features, x)
        #print("DIM F_ATTENTION ", f_attention_block.shape, "DIM F ", features.shape)
        f_attention_block += features

        

        if self.skip_connection:
            x = torch.concatenate((i_attention_block, f_attention_block, self.self_attention(x), features), dim=-1)
        else:
            x = torch.concatenate((i_attention_block, f_attention_block), dim=-1)

        #print(x.shape)

        x_dict = self.linear_layers(x, None, vectorize=vectorize)

        return x_dict
    






class SelfAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, hidden_size, dropout_rate):
        super(SelfAttentionBlock, self).__init__()

        
        mlp_dim = 512

        self.transformer = Transformer(dim=input_dim, depth=num_layers, heads=num_heads, dim_head=hidden_size // num_heads, mlp_dim=mlp_dim, dropout=dropout_rate)
        self.norm = nn.LayerNorm(patch_embedding_hidden_size)

    def forward(self, x):
        x = self.transformer(x)
        return self.norm(x)
