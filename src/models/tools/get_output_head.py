import torch
from torch import nn
from src.models.tools.model_summary import get_model_summary
from src.models.output_heads.MultiToxOutputHead import MultiToxOutputHead



# NOTE: TODO: move Flatten() into the output head class ?
def get_output_head(config, n_features, feature_map_dim_after_encoder = None):
    """
    Get the output head (linear layers) of a model by name.
    """

    output_head_name = config['model']['output_head']['name'].lower()
     
    if output_head_name == "multitox":
        output_head = MultiToxOutputHead(config=config, n_features=n_features)
    
    # TODO: add more output heads here (i.e. multi-time)
    
    else:
        raise ValueError(f"Output head {output_head_name} not recognized")
        
    return output_head
    


# def get_linear_layers(config, n_features):
#     """
#     Get the output head (linear layers) of a model by name.
#     """

#     model_name = config['model']['model_name'].lower()
#     linear_layers_name = config['model']['output_head']['name'].lower()
        
#     # linear
#     if linear_layers_name == "multitoxoutputhead":
#         linear_layers = MultiToxOutputHead(config=config, n_features=n_features)
#         flatten = nn.Flatten()
