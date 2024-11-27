import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from functools import reduce
from operator import __add__

class conv_block(torch.nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 use_bias=False):
        super(conv_block, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)

        self.pad = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=pad_value)
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1,
                                     bias=use_bias)
        self.norm2 = torch.nn.InstanceNorm3d(filters)
        self.use_activation = use_activation
        self.activation2 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)

        x = self.norm1(x)
        x = self.activation1(x)
        x = self.pad(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_activation:
            x = self.activation2(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv3d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class conv3d_padding_same(torch.nn.Module):
    """
    Padding so that the next Conv3d layer outputs an array with the same dimension as the input.
    Depth, height and width are the kernel dimensions.

    Example:
    import torch
    from functools import reduce
    from operator import __add__

    batch_size = 8
    in_channels = 3
    out_channel = 16
    kernel_size = (2, 3, 5)
    stride = 1  # could also be 2, or 3, etc.
    pad_value = 0
    conv = torch.nn.Conv3d(in_channels, out_channel, kernel_size, stride=stride)

    x = torch.empty(batch_size, in_channels, 100, 100, 100)
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    y = torch.nn.functional.pad(x, conv_padding, 'constant', pad_value)

    out = conv(y)
    print(out.shape): torch.Size([8, 16, 100, 100, 100])

    Source: https://stackoverflow.com/questions/58307036/is-there-really-no-padding-same-option-for-pytorchs-conv2d
    Source: https://pytorch.org/docs/master/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """

    def __init__(self, depth, height, width, pad_value):
        super(conv3d_padding_same, self).__init__()
        self.kernel_size = (depth, height, width)
        self.pad_value = pad_value

    def forward(self, x):
        # Determine amount of padding
        # Internal parameters used to reproduce Tensorflow "Same" padding.
        # For some reasons, padding dimensions are reversed wrt kernel sizes.
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        x_padded = torch.nn.functional.pad(x, conv_padding, 'constant', self.pad_value)

        return x_padded


class depthwise_separable_conv(torch.nn.Module):
    """
    Note: apply padding (if necessary) before applying depthwise_separable_conv().

    Source: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch/blob/master/DepthwiseSeparableConvolution/DepthwiseSeparableConvolution.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = torch.nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                         stride=stride, groups=in_channels, bias=bias)
        self.pointwise = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                         bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Output(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Output, self).__init__(in_features=in_features, out_features=out_features, bias=bias)