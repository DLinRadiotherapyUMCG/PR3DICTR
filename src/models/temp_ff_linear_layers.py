# -*- coding: utf-8 -*-
"""
This is a working file to experiment with feed forward multi-head linear layers 



"""

import torch
from torch import nn

class MultiToxOutputHead(torch.nn.Module):
    """
    A class to define the linear layers and output head(s) of the model. 
    Can be given to different models as a module, so that the different models can share the same output head (and so that the code for it is shared).
    """
    def __init__(self, config, n_features):
        super(MultiToxOutputHead, self).__init__()


        self.config = config
        self.predict_CT_contrast = False
        self.endpoint_list = [x for x in config['columns']['label'] if "CT+C" not in x]
        self.n_features = n_features
        self.dropout_p = config['model']['dropout_p'] 
        self.use_bias = config['model']['use_bias']
        self.lrelu_alpha = config['model']['lrelu_alpha']
        self.linear_units = config['model']['linear_units']
        self.clinical_variables_linear_units = config['model']['clinical_variables_linear_units']
        self.linear_units_endpoint = config['model']['linear_units_endpoint']
        self.clinical_variables_position = config['model']['clinical_variables_position']
        self.seq = config['model']['sequential']
        self.num_ohe_classes = config['model']['num_ohe_classes'] 
        self.variance_logit_head = False

        if self.clinical_variables_position >= 0 and self.clinical_variables_linear_units is not None and self.linear_units is None:
            raise ValueError('clinical_variables_position >= 0, clinical_variables_linear_units is None, and '
                             'linear_units is None is not allowed, because clinical_variables_position >= 0 implies '
                             'that the clinical variables will be concatenated linear_units[clinical_variables_position].')
        if self.clinical_variables_position <= 0 and  self.clinical_variables_linear_units is not None:
            raise ValueError('clinical_variables_position 0 (or lower)! It\'s indexing must start at 1 (for the first linear layer)!')
        if self.clinical_variables_position > len(self.linear_units)+1:
            raise ValueError('clinical_variables_position is higher than the number of linear layers + 1! \n'
                             f'clinical_variables_position = {self.clinical_variables_position}, linear layers = {len(self.linear_units)}')
        
        self._make_clinical_fc_layers(n_features)
        # makes:
        #   self.clinical_variables_fc_layer
        #   self.clinical_variables_linear_units

        self._make_shared_fc_layers()
        # makes:
        #   self.shared_fc_layers
        #   self.n_sublayers_per_linear_layer

        self._make_non_shared_endpoint_fc_layers()
        # makes:
        #   self.non_shared_endpoint_fc_layers    



    def forward(self, x, features, vectorize=False):
        x_dict = dict()

        # ----- SHARED LAYERS ----- #
        # MLP clinical variables (SHARED)
        if (self.n_features > 0) and (self.clinical_variables_linear_units is not None):
            for layer in self.clinical_variables_shared_fc_layers:
                features = layer(features)

        # Linear layers (SHARED)
        for i, layer in enumerate(self.shared_fc_layers):
            # Add features to flattened layer (clinical features are an input to the first (shared) linear layer)
            if i == 0 and self.clinical_variables_position == 1 and (self.n_features > 0):
                # Add features to flattened layer
                x = torch.cat([x, features], dim=1)

            x = layer(x)

            if (self.n_features > 0) and (
                     (i + 1) / self.n_sublayers_per_linear_layer == self.clinical_variables_position - 1) \
                     and not ( self.clinical_variables_position == 1):
                 # Add clinical features to an intermediate layer
                 x = torch.cat([x, features], dim=1)

        # Add clinical features to the final layer
        if len(self.shared_fc_layers) == self.clinical_variables_position + 1 and (self.n_features > 0):
                x = torch.cat([x, features], dim=1)
                
        # ----- NON-SHARED LAYERS, ENDPOINT SPECIFIC ----- #
        
        # Clone tensor (preserving the gradient)
        
        
        if self.seq:
            for i in range(len(self.endpoint_list)):
                if i == 0:
                    inp = x.clone()
                    x_dict[self.endpoint_list[i]] = inp.clone()
                    for layer in self.endpoint_heads[self.endpoint_list[i]]:
                        x_dict[self.endpoint_list[i]] = layer(x_dict[self.endpoint_list[i]])

                else:
                    inp = torch.cat([inp,x_dict[self.endpoint_list[i-1]]],dim=1)
                    x_dict[self.endpoint_list[i]] = inp.clone()
                    for layer in self.endpoint_heads[self.endpoint_list[i]]:
                        x_dict[self.endpoint_list[i]] = layer(x_dict[self.endpoint_list[i]])
        else:    
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


    def _make_clinical_fc_layers(self, n_features):
        
        if (self.n_features > 0) and (self.clinical_variables_linear_units is not None):
            self.clinical_variables_shared_fc_layers = torch.nn.ModuleList()

            # make a list of linear layers, starting with the clinical input size
            self.clinical_variables_linear_units = [n_features] + self.clinical_variables_linear_units

            # iteratively add linear layers to the ModuleList()
            for i in range(len(self.clinical_variables_linear_units) - 1):
                if self.dropout_p > 0:
                    self.clinical_variables_shared_fc_layers.add_module(
                        f'Clinical_shared_dropout_{i+1}',
                        torch.nn.Dropout(self.dropout_p)
                    )
                self.clinical_variables_shared_fc_layers.add_module(
                    f'Clinical_shared_linear_{i+1}',
                    torch.nn.Linear(in_features=self.clinical_variables_linear_units[i],
                                    out_features=self.clinical_variables_linear_units[i + 1],
                                    bias=self.use_bias)
                )
                self.clinical_variables_shared_fc_layers.add_module(
                    f'Clinical_shared_LReLU_{i+1}', nn.LeakyReLU(negative_slope = self.lrelu_alpha))
        else:
            # no extra layers, just feed the features directly into the model
            self.clinical_variables_linear_units = [n_features]

    def _make_shared_fc_layers(self):
        self.shared_fc_layers = torch.nn.ModuleList()
        #self.linear_units = self.linear_units
        
        for i in range(0, len(self.linear_units)):
            # `- 1` because the input of the very first fully-connected layer is from the flatten layer (instead of a linear layer).
            if i == self.clinical_variables_position - 1:
                additional_units = self.clinical_variables_linear_units[-1]
            else:
                additional_units = 0

            if self.dropout_p > 0:
                self.shared_fc_layers.add_module(f'Dropout_shared_{i+1}', torch.nn.Dropout(self.dropout_p))
            
            if i == 0:
                self.shared_fc_layers.add_module(f'Linear_shared_{i+1}',
                                                torch.nn.LazyLinear(out_features=self.linear_units[i], bias=self.use_bias))
            else:
                self.shared_fc_layers.add_module(f'Linear_shared_{i+1}',
                                            torch.nn.Linear(in_features=self.linear_units[i-1] + additional_units,
                                                            out_features=self.linear_units[i],
                                                            bias=self.use_bias))
            self.shared_fc_layers.add_module(f'LReLU_shared_{i+1}', nn.LeakyReLU(negative_slope = self.lrelu_alpha))

        if self.n_features > 0:
            self.n_sublayers_per_linear_layer = len(self.shared_fc_layers) / (len(self.linear_units))

    def _make_non_shared_endpoint_fc_layers(self):
        # Initialize linear layers (NON-SHARED, endpoint specific)
        linear_layers_endpoint_dict_i = dict()
        for endpoint in self.endpoint_list:
            linear_layers_endpoint_dict_i[endpoint] = torch.nn.ModuleList()
        self.endpoint_heads = torch.nn.ModuleDict(linear_layers_endpoint_dict_i)

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
            self.endpoint_heads[endpoint].add_module('Output_{}'.format(endpoint),
                                torch.nn.LazyLinear(out_features=self.num_ohe_classes, bias=self.use_bias))
        
    def _make_sequential_output_head(self):
        # Initialize linear layers (NON-SHARED, endpoint specific) extended to include the previous output as new input
        linear_layers_endpoint_dict_i = dict()
        for endpoint in self.endpoint_list:
            linear_layers_endpoint_dict_i[endpoint] = torch.nn.ModuleList()
        self.endpoint_heads = torch.nn.ModuleDict(linear_layers_endpoint_dict_i)

        for endpoint in self.endpoint_list:
            # loop through the non-shared linear layers
            for i in range(len(self.linear_units_endpoint)):
                if self.dropout_p > 0:
                    self.endpoint_heads[endpoint].add_module(f'Endpoint_Dropout_{i+1}', torch.nn.Dropout(self.dropout_p))
                if i == 0:
                    self.endpoint_heads[endpoint].add_module(f'Endpoint_Linear_{i+1}',
                                                    torch.nn.LazyLinear(out_features=self.linear_units_endpoint[i] + i, bias=self.use_bias))
                else:
                    self.endpoint_heads[endpoint].add_module(f'Endpoint_Linear_{i+1}',
                                         torch.nn.Linear(in_features=self.linear_units_endpoint[i-1] + i,
                                                         out_features=self.linear_units_endpoint[i] + i, bias=self.use_bias))
                self.endpoint_heads[endpoint].add_module(f'Endpoint_LReLU_{i+1}', nn.LeakyReLU(negative_slope = self.lrelu_alpha))
            # output layer/head for this toxicity endpoint
            self.endpoint_heads[endpoint].add_module('Output_{}'.format(endpoint),
                                torch.nn.LazyLinear(out_features=self.num_ohe_classes, bias=self.use_bias))
        
        