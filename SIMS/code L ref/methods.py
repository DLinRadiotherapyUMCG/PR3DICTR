# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:53:31 2024

@author: HoekL02
"""



#%% Models, DenseNet, EfficientNet, Densenet

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math
from functools import partial
from layers import *

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_C,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 n_features,
                 num_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.n_features = n_features
        self.conv1 = nn.Conv3d(
            sample_input_C,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = Output(512 * block.expansion + self.n_features, num_classes)
        # self.fc.__class__.__name__ = 'Output'

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Add features
        if self.n_features > 0:
            x = torch.cat([x, features], dim=1)

        x = self.fc(x)

        return x
    
class EffNetV2(nn.Module):
    """
    t: expand ratio in (Fused-)MBConv block
    c: output channels
    n: number of layers
    s: stride
    use_se: use Fused-MBConv block if True, else use MBConv block

    """
    def __init__(self, cfgs, n_features,
                 dropout_p,
                 linear_units,
                 clinical_variables_position,
                 clinical_variables_linear_units,
                 clinical_variables_dropout_p,
                 use_bias,
                 lrelu_alpha,
                 channels=3, num_classes=2, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs
        self.n_features = n_features
        self.clinical_variables_position = clinical_variables_position
        self.clinical_variables_linear_units = clinical_variables_linear_units
        self.use_bias = use_bias
        self.lrelu_alpha = lrelu_alpha

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(channels, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        # output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self._initialize_weights()

        # Initialize MLP layers for clinical variables
        if (self.n_features > 0) and (clinical_variables_linear_units is not None):
            self.clinical_variables_layers = torch.nn.ModuleList()
            clinical_variables_linear_units = [n_features] + clinical_variables_linear_units
            for i in range(len(clinical_variables_linear_units) - 1):
                self.clinical_variables_layers.add_module(
                    'dropout%s' % i,
                    torch.nn.Dropout(clinical_variables_dropout_p[i])
                )
                self.clinical_variables_layers.add_module(
                    'linear%s' % i,
                    torch.nn.Linear(in_features=clinical_variables_linear_units[i],
                                    out_features=clinical_variables_linear_units[i + 1],
                                    bias=use_bias)
                )
                self.clinical_variables_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))
        else:
            clinical_variables_linear_units = [n_features]

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        if (linear_units is None) or (len(linear_units) == 0):
            self.out_layer = Output(in_features=output_channel + clinical_variables_linear_units[-1], out_features=num_classes,
                                    bias=use_bias)
        else:
            linear_units = [output_channel] + linear_units
            for i in range(len(linear_units) - 1):
                self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))

                if self.clinical_variables_position + 1 == i:
                    self.linear_layers.add_module('linear%s' % i,
                                                  torch.nn.Linear(in_features=linear_units[i] + clinical_variables_linear_units[-1],
                                                                  out_features=linear_units[i + 1], bias=use_bias))
                else:
                    self.linear_layers.add_module('linear%s' % i,
                                                  torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i + 1],
                                                                  bias=use_bias))

                self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))
            self.n_sublayers_per_linear_layer = len(self.linear_layers) / (len(linear_units) - 1)

            # Initialize output layer
            if (self.n_features > 0) and (clinical_variables_linear_units is not None):
                # Add additional input units to the output layer if MLP output is concatenated at the last linear layer
                if self.clinical_variables_position + 1 == len(linear_units) - 1:
                    self.out_layer = Output(in_features=linear_units[-1] + clinical_variables_linear_units[-1],
                                            out_features=num_classes, bias=use_bias)
                else:
                    self.out_layer = Output(in_features=linear_units[-1], out_features=num_classes, bias=use_bias)
            else:
                self.out_layer = Output(in_features=linear_units[-1] + self.n_features, out_features=num_classes,
                                        bias=use_bias)
            # self.out_layer.__class__.__name__ = 'Output'


    def forward(self, x, features):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # MLP clinical variables
        if (self.n_features > 0) and (self.clinical_variables_linear_units is not None):
            for layer in self.clinical_variables_layers:
                features = layer(features)

        if (self.n_features > 0) and (len(self.linear_layers) == 0):
            # Add features to flattened layer
            x = torch.cat([x, features], dim=1)
        else:
            # Linear layers
            for i, layer in enumerate(self.linear_layers):
                x = layer(x)
                if (self.n_features > 0) and (
                        (i + 1) / self.n_sublayers_per_linear_layer == self.clinical_variables_position + 1):
                    # Add features to a linear layer
                    x = torch.cat([x, features], dim=1)

        x = self.out_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                receptive_field_size = 1
                for s in m.kernel_size:
                    receptive_field_size *= s

                n = m.out_channels * receptive_field_size
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
                
class DCNN_LReLU(torch.nn.Module):
    """
    Deep CNN
    """

    def __init__(self, n_input_channels, depth, height, width, n_features, num_classes, filters, kernel_sizes, strides,
                 pad_value, n_down_blocks, lrelu_alpha, dropout_p, pooling_conv_filters, perform_pooling,
                 linear_units,
                 clinical_variables_position,
                 clinical_variables_linear_units,
                 clinical_variables_dropout_p,
                 use_bias=False):
        super(DCNN_LReLU, self).__init__()
        self.n_features = n_features
        self.pooling_conv_filters = pooling_conv_filters
        self.perform_pooling = perform_pooling
        self.clinical_variables_position = clinical_variables_position
        self.clinical_variables_linear_units = clinical_variables_linear_units

        # Determine Linear input channel size
        end_depth = math.ceil(depth / (2 ** n_down_blocks[0]))
        end_height = math.ceil(height / (2 ** n_down_blocks[1]))
        end_width = math.ceil(width / (2 ** n_down_blocks[2]))

        # Initialize conv blocks
        in_channels = [n_input_channels] + list(filters[:-1])
        self.conv_blocks = torch.nn.ModuleList()
        for i in range(len(in_channels)):
            use_activation = True
            self.conv_blocks.add_module('conv_block%s' % i,
                                        conv_block(in_channels=in_channels[i], filters=filters[i],
                                                   kernel_size=kernel_sizes[i], strides=strides[i],
                                                   pad_value=pad_value, lrelu_alpha=lrelu_alpha,
                                                   use_activation=use_activation, use_bias=use_bias))

        # Initialize pooling conv
        if self.pooling_conv_filters is not None:
            pooling_conv_kernel_size = [end_depth, end_height, end_width]
            self.pool = pooling_conv(in_channels=filters[-1], filters=pooling_conv_filters,
                                     kernel_size=pooling_conv_kernel_size, strides=1,
                                     lrelu_alpha=lrelu_alpha, use_bias=use_bias)
            end_depth, end_height, end_width = 1, 1, 1
            filters[-1] = self.pooling_conv_filters
        elif self.perform_pooling:
            # self.pool = torch.nn.AvgPool3d(kernel_size=(1, end_height, end_width))
            # end_depth, end_height, end_width = depth, 1, 1
            # self.pool = torch.nn.AvgPool3d(kernel_size=(end_depth, end_height, end_width))
            self.pool = torch.nn.MaxPool3d(kernel_size=(end_depth, end_height, end_width))
            end_depth, end_height, end_width = 1, 1, 1

        # Initialize flatten layer
        self.flatten = torch.nn.Flatten()

        # Initialize MLP layers for clinical variables
        if (self.n_features > 0) and (clinical_variables_linear_units is not None):
            self.clinical_variables_layers = torch.nn.ModuleList()
            clinical_variables_linear_units = [n_features] + clinical_variables_linear_units
            for i in range(len(clinical_variables_linear_units) - 1):
                self.clinical_variables_layers.add_module(
                    'dropout%s' % i,
                    torch.nn.Dropout(clinical_variables_dropout_p[i])
                )
                self.clinical_variables_layers.add_module(
                    'linear%s' % i,
                    torch.nn.Linear(in_features=clinical_variables_linear_units[i],
                                    out_features=clinical_variables_linear_units[i + 1],
                                    bias=use_bias)
                )
                self.clinical_variables_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))
        else:
            clinical_variables_linear_units = [n_features]

        # Initialize linear layers
        self.linear_layers = torch.nn.ModuleList()
        linear_units = [end_depth * end_height * end_width * filters[-1]] + linear_units
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module('dropout%s' % i, torch.nn.Dropout(dropout_p[i]))
            self.linear_layers.add_module('linear%s' % i,
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i+1],
                                                          bias=use_bias))
            self.linear_layers.add_module('lrelu%s' % i, torch.nn.LeakyReLU(negative_slope=lrelu_alpha))
        self.n_sublayers_per_linear_layer = len(self.linear_layers) / (len(linear_units) - 1)

        # Initialize output layer
        if (self.n_features > 0) and (clinical_variables_linear_units is not None):
            # Add additional input units to the output layer if MLP output is concatenated at the last linear layer
            if self.clinical_variables_position + 1 == len(linear_units) - 1:
                self.out_layer = Output(in_features=linear_units[-1] + clinical_variables_linear_units[-1],
                                        out_features=num_classes, bias=use_bias)
            else:
                self.out_layer = Output(in_features=linear_units[-1], out_features=num_classes, bias=use_bias)
        else:
            self.out_layer = Output(in_features=linear_units[-1] + self.n_features, out_features=num_classes,
                                    bias=use_bias)
        # self.out_layer.__class__.__name__ = 'Output'

    def forward(self, x, features):
        # Blocks
        for block in self.conv_blocks:
            x = block(x)

        # Pooling layers
        if (self.pooling_conv_filters is not None) or self.perform_pooling:
            x = self.pool(x)

        x = self.flatten(x)

        # MLP clinical variables
        if (self.n_features > 0) and (self.clinical_variables_linear_units is not None):
            for layer in self.clinical_variables_layers:
                features = layer(features)

        # Linear layers
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if (self.n_features > 0) and (
                    (i + 1) / self.n_sublayers_per_linear_layer == self.clinical_variables_position + 1):
                # Add features
                x = torch.cat([x, features], dim=1)

        # Output
        x = self.out_layer(x)

        return x
    
    import math

import torch
from torch.optim.optimizer import Optimizer

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

__all__ = ('AdaBound',)


class AdaBound(Optimizer):
    r"""Implements AdaBound algorithm.

    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of
    Learning Rate`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        final_lr: final (SGD) learning rate (default: 0.1)
        gamma: convergence speed of the bound functions
            (default: 1e-3)
        eps: term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsbound: whether to use the AMSBound variant of this algorithm

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AdaBound(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1902.09843

    Note:
        Reference code: https://github.com/Luolc/AdaBound
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsbound: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if final_lr < 0.0:
            raise ValueError(
                'Invalid final learning rate: {}'.format(final_lr)
            )
        if not 0.0 <= gamma < 1.0:
            raise ValueError('Invalid gamma parameter: {}'.format(gamma))
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super(AdaBound, self).__init__(params, defaults)
        self.base_lrs = [group['lr'] for group in self.param_groups]

    def __setstate__(self, state: State) -> None:
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'AdaBound does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsbound:
                        # Maintains max of all exp. moving avg. of
                        # sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running
                    # avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = (
                    group['lr']
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround
                # to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (
                    1 - 1 / (group['gamma'] * state['step'] + 1)
                )
                upper_bound = final_lr * (
                    1 + 1 / (group['gamma'] * state['step'])
                )
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(
                    exp_avg
                )

                p.data.add_(-step_size)
        return loss