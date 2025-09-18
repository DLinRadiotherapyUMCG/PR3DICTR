import torch
from torch import nn
from src.models.tools.model_summary import get_model_summary
from src.models.output_heads.MultiLabelOutputHead import MultiLabelOutputHead

# NOTE: TODO: move Flatten() into the output head class ?
def get_output_head(config, n_features, feature_map_dim_after_encoder = None):
    """
    Get the output head (linear layers) of a model by name.
    """

    output_head_name = config['model']['output_head']['name'].lower()
     
    if output_head_name == "multilabel":
        output_head = MultiLabelOutputHead(config=config, n_features=n_features)
    
    
    # TODO: add more output heads here (i.e. multi-time? )
    
    else:
        raise ValueError(f"Output head {output_head_name} not recognized")
        
    return output_head
    