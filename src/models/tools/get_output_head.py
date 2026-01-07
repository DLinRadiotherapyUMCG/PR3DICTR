from src.models.output_heads.MultiLabelOutputHead import MultiLabelOutputHead

def get_output_head(config, n_features, feature_map_dim_after_encoder = None):
    """
    Get the output head (linear layers) of a model by name.
    """

    output_head_name = config['model']['output_head']['name'].lower()
     
    if output_head_name == "multilabel":
        output_head = MultiLabelOutputHead(config=config, n_features=n_features)
    
    else:
        raise ValueError(f"Output head {output_head_name} not recognized")
        
    return output_head
    