import torch
from torch import nn


class HybridFusionOutputHeads(torch.nn.Module):
    """
    A class to define the linear layers and output head(s) of the model. 
    Can be given to different models as a module, so that the different models can share the same output head (and so that the code for it is shared).
    """
    def __init__(self, config, ):
        super(HybridFusionOutputHeads, self).__init__()

        self.endpoint_list =config['columns']['labels']
        
        self.dropout_p = config['model']['dropout_p'] 
        self.num_ohe_classes = config['model']['num_ohe_classes'] 
        self.use_bias = config['model']['use_bias']
        self.lrelu_alpha = config['model']['lrelu_alpha']
        self.linear_units_endpoint = config['model']['linear_units_endpoint']
        
        self.variance_logit_head = False

        #self.flatten = nn.Flatten()
        self._make_output_heads()
        


    def forward(self, cls_token, image_representation, features=None, vectorize=False):

        # Flatten the input tensor
        #x = self.flatten(x)

        x_dict = dict()

        x = (cls_token, image_representation)        

        # ----- NON-SHARED LAYERS, ENDPOINT SPECIFIC ----- #
        # Clone tensor (preserving the gradient)
        for endpoint in self.endpoint_list:
            x_dict[endpoint] = x# .clone()
        
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



    
    def _make_output_heads(self):

        # Initialize linear layers (NON-SHARED, endpoint specific)
        linear_layers_endpoint_dict_i = dict()
        for endpoint in self.endpoint_list:
            linear_layers_endpoint_dict_i[endpoint] = torch.nn.ModuleList()
        self.endpoint_heads = torch.nn.ModuleDict(linear_layers_endpoint_dict_i)

        for endpoint in self.endpoint_list:
             
            self.endpoint_heads[endpoint].add_module(f'Input_Weighting', ClassSpecificInputWeighting())


            # loop through the non-shared linear layers
            for i in range(len(self.linear_units_endpoint)):
                if self.dropout_p > 0:
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
            if self.dropout_p > 0:
                    self.endpoint_heads[endpoint].add_module(f'Output_{endpoint}_Dropout', torch.nn.Dropout(self.dropout_p))
            self.endpoint_heads[endpoint].add_module(f'Output_{endpoint}',
                                torch.nn.LazyLinear(out_features=self.num_ohe_classes, bias=self.use_bias))






class ClassSpecificInputWeighting(nn.Module):
    
    def __init__(self):
        super(ClassSpecificInputWeighting, self).__init__()
        # Learnable alpha for each class
        self.alpha =  nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, tokens_tuple):
        (cls_token, image_tokens) = tokens_tuple

        #print(cls_token[0])

        # Use per-class alpha (reshaped for broadcasting)
        combined_representation = self.alpha * cls_token + (1 - self.alpha) * image_tokens

        #print(self.alpha.view(1, -1).shape)

        #print(self.alpha.shape)

        return combined_representation
