import torch
import numpy as np
from einops import rearrange, repeat
import time

from src.models.ViT import ViT



class HIPT(torch.nn.Module):
    """
    Hierarchical ViT
    """

    def __init__(self, channels, depth, height, width) -> None:
        super().__init__()

        self.patch_model = None
        self.region_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.region_block_size = 8
        self.N_region_blocks = (
                                depth // self.region_block_size,  # number of patches in each dimension (for the final ViT)
                                height // self.region_block_size, 
                                width // self.region_block_size
                                  )

        ps = 2
        ps_global = 2

        region_vector_dim = 512
        self.region_model = ViT(image_size=(self.region_block_size, self.region_block_size, self.region_block_size), patch_size=(ps,ps,ps),
                    dim=region_vector_dim, depth=4, 
                    heads=12, mlp_dim=1024, pool = 'cls', 
                    channels = channels, dim_head = 256, emb_dropout = 0.1,
                    dropout = 0.1) 
        
        # find the number of total blocks used in the final ViT
        total_N_blocks = self.N_region_blocks[0] * self.N_region_blocks[1] * self.N_region_blocks[2]

        #print(self.N_region_blocks, total_N_blocks)

        self.global_model = ViT(image_size=self.N_region_blocks, patch_size=(ps_global,ps_global,ps_global),
                    dim=512, depth=4, 
                    heads=24, mlp_dim=1024, pool = 'cls', 
                    channels = region_vector_dim, dim_head = 256, emb_dropout = 0.1,
                    dropout = 0.1) 


    
    def forward(self, x, autoencoder=False):
        """
        Subdivide the image into 8x8x8 cubes
        """

        # subdivide the image into 'patch_block_size'-sized cubes
        b = self.region_block_size
        x_blocks = x.unfold(2, b, b).unfold(3, b, b).unfold(4, b, b)         # [batch_size, channels, h, w, d] -> [batch_size, channels, h/b, w/b, d/b, b, b, b]

        x_blocks = rearrange(x_blocks, 
            'batch c n1 n2 n3 b1 b2 b3 -> batch (n1 n2 n3) c b1 b2 b3')   # [batch_size, channels, h/b, w/b, d/b, b, b, b] -> [batch_size, (h/b * w/b * d/b), channels, b, b, b] 
        
        region_features = []
        for mini_bs_idx in range(x_blocks.shape[1]):
            x_block = x_blocks[:, mini_bs_idx].to(self.device, non_blocking=True)
            
            #temp_out = self.region_model(x_block)#.unsqueeze(1)              # model outputs = [batch_size, n_patches, vit_dim]
            region_features.append(self.region_model(x_block).detach().cpu())

        # stack together all of the region features (one vector per region)
        t0 = time.time()
        region_features = torch.stack(region_features, dim=1)
        #print("stacking time", time.time()-t0)

        # reshape, the number of patches in each dimension --> feature map dimensions (n1, n2, n3)
        (n1, n2, n3) = self.N_region_blocks
        region_features = rearrange(region_features, 'batch (n1 n2 n3) v  -> batch v n1 n2 n3', n1=n1, n2=n2, n3=n3)  # [batch_size, n_patches, vit_dim] -> [batch_size, vit_dim, h/b, w/b, d/b]
        #print(region_features.shape)
        region_features = region_features.to(self.device, non_blocking=True)
        # global-level ViT model
        features_out = self.global_model(region_features)

        return features_out