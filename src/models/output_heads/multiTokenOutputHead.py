import torch
from torch import nn


class multiTokenOutputHead(torch.nn.Module):
    """
    A class to define the linear layers and output head(s) of the model. 
    Can be given to different models as a module, so that the different models can share the same output head (and so that the code for it is shared).
    """
    def __init__(self, config):
        super(multiTokenOutputHead, self).__init__()

        self.predict_CT_contrast = False
        self.endpoint_list = [x for x in config['columns']['labels'] if "CT+C" not in x]
        self.dropout_p = config['model']['dropout_p'] 
        self.num_ohe_classes = config['model']['num_ohe_classes'] 
        self.use_bias = config['model']['use_bias']
        self.lrelu_alpha = config['model']['lrelu_alpha']
        #self.clinical_variables_linear_units = config['model']['clinical_variables_linear_units']
        self.linear_units_endpoint = config['model']['linear_units_endpoint']
        #self.clinical_variables_position = config['model']['clinical_variables_position']

        self.variance_logit_head = False


        self._make_non_shared_endpoint_fc_layers()
        # makes:
        #   self.non_shared_endpoint_fc_layers    

        if self.predict_CT_contrast:
            self._make_CT_contrast_output_head()
            # makes:
            #   self.CT_contrast_output_head


    def forward(self, x, features=None, vectorize=False):

        # Flatten the input tensor
        #x = self.flatten(x)

        x_dict = dict()

        # ----- NON-SHARED LAYERS, ENDPOINT SPECIFIC ----- #
        # Clone tensor (preserving the gradient)
        #print(x.shape)
        for idx, endpoint in enumerate(self.endpoint_list):

            x_dict[endpoint] = x[:, idx, :]

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



    
    def _make_non_shared_endpoint_fc_layers(self):
        # Initialize linear layers (NON-SHARED, endpoint specific)
        linear_layers_endpoint_dict_i = dict()
        for endpoint in self.endpoint_list:
            linear_layers_endpoint_dict_i[endpoint] = torch.nn.ModuleList()
        self.endpoint_heads = torch.nn.ModuleDict(linear_layers_endpoint_dict_i)

        # Add additional input units for the first non-shared linear layers if the clinical variables are concatenated        
        # if self.clinical_variables_position - 1 == len(self.linear_units):
        #     self.linear_units_endpoint = [self.linear_units[-1] + self.clinical_variables_linear_units[-1]] + self.linear_units_endpoint
        # else:
        #     self.linear_units_endpoint = [self.linear_units[-1]] + self.linear_units_endpoint

        for endpoint in self.endpoint_list:
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

    def _make_CT_contrast_output_head(self):
        self.CT_contrast_layers = torch.nn.ModuleList()

        self.CT_contrast_layers.add_module(f'CT_contrast_{1}',
                                                 torch.nn.LazyLinear(out_features=64, bias=self.use_bias)
                                                 )
        self.CT_contrast_layers.add_module(f'LReLU_CT_contrast_{1}', nn.LeakyReLU(negative_slope = self.lrelu_alpha))

        self.CT_contrast_layers.add_module(f'CT_contrast_{2}',
                                                torch.nn.Linear(in_features=64,
                                                                out_features=16, bias=self.use_bias))
        self.CT_contrast_layers.add_module(f'LReLU_CT_contrast_{2}', nn.LeakyReLU(negative_slope = self.lrelu_alpha))
        self.CT_contrast_layers.add_module(f'CT_contrast_{3}',
                                                torch.nn.Linear(in_features=16,
                                                                out_features=1, bias=self.use_bias))


