import torch
from torch import nn

class MultiLabelOutputHead(nn.Module):
    """
    Defines the output head for multi-label classification, supporting shared and endpoint-specific layers,
    as well as optional clinical variable integration.
    """
    def __init__(self, config, n_features):
        super().__init__()

        # Configuration
        self.endpoint_list = config['columns']['labels']
        self.n_features = n_features
        self.dropout_p = config['model']['dropout_p']
        self.num_ohe_classes = config['model']['output_head']['num_ohe_classes']
        self.use_bias = config['model']['use_bias']
        self.lrelu_alpha = config['model']['lrelu_alpha']
        self.linear_units = config['model']['output_head']['linear_units'] or None
        self.clinical_units = config['model']['output_head']['clinical_variables_linear_units']
        self.endpoint_units = config['model']['output_head']['linear_units_endpoint']
        self.clinical_pos = config['model']['output_head']['clinical_variables_position']

        # Validation
        self._validate_config()

        # Build layers
        self._build_clinical_layers(n_features)
        self._build_shared_layers()
        self._build_endpoint_layers()

    def forward(self, x, features=None, vectorize=False):
        """
        Forward pass through the output head.
        Args:
            x: Main input tensor.
            features: Clinical features tensor (optional).
            vectorize: If True, stack outputs for all endpoints into a single tensor.
        Returns:
            Dict of endpoint outputs or stacked tensor.
        """
        # Process clinical features if present
        if self.n_features > 0 and self.clinical_units is not None:
            for layer in self.clinical_layers:
                features = layer(features)

        # Shared layers with clinical feature concatenation at specified position
        for i, layer in enumerate(self.shared_layers):
            if i == 0 and self.clinical_pos == 1 and self.n_features > 0:
                x = torch.cat([x, features], dim=1)
            x = layer(x)
            if self.n_features > 0 and self._should_concat_clinical(i):
                x = torch.cat([x, features], dim=1)

        # Final clinical concat if required
        if self._should_concat_final():
            x = torch.cat([x, features], dim=1)
        elif len(self.shared_layers) == 0 and self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        # Endpoint-specific layers
        outputs = {}
        for endpoint in self.endpoint_list:
            endpoint_x = x.clone()
            for layer in self.endpoint_layers[endpoint]:
                endpoint_x = layer(endpoint_x)
            outputs[endpoint] = endpoint_x
    
        if vectorize:
            return torch.cat([outputs[ep] for ep in self.endpoint_list], dim=1)
        return outputs

    def _validate_config(self):
        # function to check that the linear layer parameters in the config are valid 
        if self.n_features > 0: # only need to check these things if there are clinical features in the first place
            if self.clinical_pos > 1 and self.linear_units is None:
                raise ValueError('clinical_variables_position cannot be more than 1 if there are no (shared) linear layers.')
            if self.clinical_pos < 1:
                raise ValueError('clinical_variables_position must be at least 1! (for the first linear layer).')
            if self.linear_units and self.clinical_pos > len(self.linear_units) + 1:
                raise ValueError('clinical_variables_position exceeds number of linear layers + 1.')

    def _build_clinical_layers(self, n_features):
        # Build linear layers for processing clinical features
        if self.n_features > 0 and self.clinical_units is not None:
            units = [n_features] + self.clinical_units
            self.clinical_layers = nn.ModuleList()
            for i in range(len(units) - 1):
                self.clinical_layers.append(nn.Dropout(self.dropout_p))
                self.clinical_layers.append(nn.Linear(units[i], units[i + 1], bias=self.use_bias))
                self.clinical_layers.append(nn.LeakyReLU(self.lrelu_alpha))
        else:
            self.clinical_layers = nn.ModuleList()

    def _build_shared_layers(self):
        # Build the shared linear layers
        self.shared_layers = nn.ModuleList()
        if not self.linear_units:
            return
        for i, out_units in enumerate(self.linear_units):
            in_units = self.linear_units[i - 1] if i > 0 else None
            if i == self.clinical_pos - 1 and self.n_features > 0:
                clinical_units = self.clinical_units[-1] if self.clinical_units else self.n_features
                in_units = (in_units or 0) + clinical_units
            self.shared_layers.append(nn.Dropout(self.dropout_p))
            if i == 0:
                self.shared_layers.append(nn.LazyLinear(out_units, bias=self.use_bias))
            else:
                self.shared_layers.append(nn.Linear(in_units, out_units, bias=self.use_bias))
            self.shared_layers.append(nn.LeakyReLU(self.lrelu_alpha))

        if self.n_features > 0 and self.linear_units:
            self.n_sublayers_per_linear_layer = len(self.shared_layers) / len(self.linear_units)

    def _build_endpoint_layers(self):
        # Build endpoint-specific layers (one set per endpoint, can be thought of as 'output heads' for each endpoint)
        self.endpoint_layers = nn.ModuleDict()
        for endpoint in self.endpoint_list:
            layers = nn.ModuleList()
            for i, out_units in enumerate(self.endpoint_units):
                layers.append(nn.Dropout(self.dropout_p))
                if i == 0:
                    layers.append(nn.LazyLinear(out_units, bias=self.use_bias))
                else:
                    layers.append(nn.Linear(self.endpoint_units[i - 1], out_units, bias=self.use_bias))
                layers.append(nn.LeakyReLU(self.lrelu_alpha))
            layers.append(nn.Dropout(self.dropout_p))
            layers.append(nn.LazyLinear(self.num_ohe_classes, bias=self.use_bias))
            self.endpoint_layers[endpoint] = layers

    def _should_concat_clinical(self, i):
        # Determines if clinical features should be concatenated at this shared layer
        return ((i + 1) / getattr(self, 'n_sublayers_per_linear_layer', 1) == self.clinical_pos - 1
                and self.clinical_pos != 1)

    def _should_concat_final(self):
        # Determines if clinical features should be concatenated after last shared layer
        return (self.linear_units and len(self.shared_layers) == self.clinical_pos + 1 and self.n_features > 0)
