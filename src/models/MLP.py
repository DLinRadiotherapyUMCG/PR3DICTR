import torch
from torch import nn





class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for processing clinical variables.
    Only gets used if there are no image inputs (in which case no image encoder is needed).
    Args:
        encoder: Not used, for compatibility with other models and the rest of the codebase
        config: Configuration dictionary containing model parameters
        n_features (int): Number of clinical features
        metadata: Not used, for compatibility with other models
    Returns:
        x_dict: Dictionary with endpoint-specific outputs
    """

    def __init__(self, config, encoder, n_features : int, metadata = None):
        super(MultiLayerPerceptron, self).__init__()

        self.endpoint_list = config['columns']['labels']
        self.n_features = n_features
        self.dropout_p = config['model']['dropout_p'] 
        self.num_ohe_classes = config['model']['output_head']['num_ohe_classes'] 
        self.use_bias = config['model']['use_bias']
        self.lrelu_alpha = config['model']['lrelu_alpha']
        self.linear_units = config['model']['output_head']['linear_units']
        self.linear_units_endpoint = config['model']['output_head']['linear_units_endpoint']
        

        self._make_shared_fc_layers()
        self._make_non_shared_endpoint_fc_layers()
    
    def forward(self, x=None, features=None, vectorize=False):

        x_dict = dict()

        # the only input is clinical features. call it 'x' just for cleaner naming
        x = features

        # ----- SHARED LAYERS ----- # 
        # Linear layers for clinical variables (SHARED)
        for i, layer in enumerate(self.shared_fc_layers):

            x = layer(x)



        # ----- NON-SHARED LAYERS, ENDPOINT SPECIFIC ----- #
        # Clone tensor (preserving the gradient)
        for endpoint in self.endpoint_list:
            x_dict[endpoint] = x.clone()
        assert len(x_dict) == len(self.endpoint_heads) or len(x_dict) - 1 == len(self.endpoint_heads)

        # Linear layers (NON-SHARED, endpoint specific)
        for endpoint in self.endpoint_list:

            for layer in self.endpoint_heads[endpoint]:
                x_dict[endpoint] = layer(x_dict[endpoint])

        if vectorize: # to stack the inputs into a single tensor (for [Captum's] attention maps)
            output_tensor = torch.cat([x_dict[endpoint] for endpoint in self.endpoint_list], dim=1)
            return output_tensor
        else:
            return x_dict
        


    def _make_shared_fc_layers(self):
        self.shared_fc_layers = torch.nn.ModuleList()

        linear_layer_sizes = [self.n_features] + self.linear_units if len(self.linear_units) > 0 else [self.n_features]
        
        if len(linear_layer_sizes) == 1:
            return # no shared layers to make

        for i in range(0, len(linear_layer_sizes) - 1):
            
            self.shared_fc_layers.add_module(f'Dropout_shared_{i+1}', torch.nn.Dropout(self.dropout_p))
            
            self.shared_fc_layers.add_module(f'Linear_shared_{i+1}',
                                            torch.nn.Linear(in_features=linear_layer_sizes[i],
                                                            out_features=linear_layer_sizes[i+1],
                                                            bias=self.use_bias))
            
            self.shared_fc_layers.add_module(f'LReLU_shared_{i+1}', nn.LeakyReLU(negative_slope = self.lrelu_alpha))


    
    def _make_non_shared_endpoint_fc_layers(self):
        # Initialize linear layers (NON-SHARED, endpoint specific)
        linear_layers_endpoint_dict_i = dict()
        for endpoint in self.endpoint_list:
            linear_layers_endpoint_dict_i[endpoint] = torch.nn.ModuleList()
        self.endpoint_heads = torch.nn.ModuleDict(linear_layers_endpoint_dict_i)


        for endpoint in self.endpoint_list:
            # loop through the non-shared linear layers
            for i in range(len(self.linear_units_endpoint)):
                
                self.endpoint_heads[endpoint].add_module(f'Endpoint_Dropout_{i+1}', torch.nn.Dropout(self.dropout_p))

                if i == 0:
                    self.endpoint_heads[endpoint].add_module(f'Endpoint_Linear_{i+1}',
                                                    torch.nn.LazyLinear(out_features=self.linear_units_endpoint[i], bias=self.use_bias))
                else:
                    self.endpoint_heads[endpoint].add_module(f'Endpoint_Linear_{i+1}',
                                         torch.nn.Linear(in_features=self.linear_units_endpoint[i-1],
                                                         out_features=self.linear_units_endpoint[i], bias=self.use_bias))
                self.endpoint_heads[endpoint].add_module(f'Endpoint_LReLU_{i+1}', nn.LeakyReLU(negative_slope = self.lrelu_alpha))
            # output layer/head for this toxicity endpoint
            
            self.endpoint_heads[endpoint].add_module(f'Output_{endpoint}_Dropout', torch.nn.Dropout(self.dropout_p))

            self.endpoint_heads[endpoint].add_module(f'Output_{endpoint}',
                                torch.nn.LazyLinear(out_features=self.num_ohe_classes, bias=self.use_bias))



    
