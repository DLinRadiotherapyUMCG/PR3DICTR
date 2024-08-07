# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py
import torch
import torch.nn as nn
import math
from models.linear_layers import MultiToxOutputHead



def conv3x3(in_planes, out_planes, groups=1, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    # expansion = 2
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32, lrelu_alpha=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_alpha, inplace=True)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, groups=num_group)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.lrelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32, lrelu_alpha=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes * 2)
        self.conv2 = nn.Conv3d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm3d(planes * 2)
        self.conv3 = nn.Conv3d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_alpha, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.lrelu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, config, block, layers, n_features,
                 use_bias,
                 input_channels=3, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.n_features = n_features
        self.use_bias = use_bias
        kernel_sizes = config.kernel_sizes
        strides = config.strides
        lrelu_alpha = config.lrelu_alpha

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=kernel_sizes[0], stride=strides[0], padding=kernel_sizes[0]//2,
                              bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_alpha, inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0], num_group, stride=2)
        # self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)

        in_channels = [64, 128, 256, 512]

        self.backbone = torch.nn.ModuleList()
        for idx in range(1, len(layers)):
            layer = self._make_layer(block, in_channels[idx-1], layers[idx],
                                       num_group, stride=2)
            
            self.backbone.add_module(f'block%s' % idx, layer)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        

        #self.output_head = MultiToxOutputHead(config=config, n_features=n_features)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, num_group, stride=1, lrelu_alpha=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group, lrelu_alpha=lrelu_alpha))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group, lrelu_alpha=lrelu_alpha))

        return nn.Sequential(*layers)


    def forward(self, x, autoencoder=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.maxpool(x)

        for layer in self.backbone:
            x = layer(x)

        if autoencoder == False:
            x = self.avgpool(x)
        #x = x.view(x.size(0), -1)  # flatten

        return x
    



def get_resnext_relu(config, model_depth, channels, n_features):
    assert model_depth in [10, 18, 34, 50, 101, 152]

    use_bias = config.use_bias

    if model_depth == 10:
        model = ResNeXt(config, BasicBlock, [1, 1, 1, 1], input_channels=channels, n_features=n_features,
                        use_bias=use_bias,
                        )
    elif model_depth == 18:
        model = ResNeXt(config, BasicBlock, [2, 2, 2, 2], input_channels=channels, n_features=n_features,
                        use_bias=use_bias,
                        )
    elif model_depth == 34:
        model = ResNeXt(config, BasicBlock, [3, 4, 6, 3], input_channels=channels, n_features=n_features,
                        use_bias=use_bias,
                        )
    elif model_depth == 50:
        model = ResNeXt(config, Bottleneck, [3, 4, 6, 3], input_channels=channels, n_features=n_features,
                        use_bias=use_bias,
                        )
    elif model_depth == 101:
        model = ResNeXt(config, Bottleneck, [3, 4, 23, 3], input_channels=channels, n_features=n_features,
                        use_bias=use_bias,
                        )
    elif model_depth == 152:
        model = ResNeXt(config, Bottleneck, [3, 8, 36, 3], input_channels=channels, n_features=n_features,
                        use_bias=use_bias,
                        )
    else:
        raise "Invalid ResNeXt model depth !! "

    return model

