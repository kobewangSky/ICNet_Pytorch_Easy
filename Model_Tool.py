import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias = True, dilation = 1, is_batchnorm = True):
        super(Conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        if is_batchnorm:
            self.model = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.model = nn.Sequential(conv_mod)

    def forward(self, input):
        output = self.model(input)
        return output

class DeConv2DBatchNorm(nn.Module):
    def __init__(self, in_channel, n_filter, k_sizes, stride, padding, bias = True):
        super(DeConv2DBatchNorm, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(int(in_channel), int(n_filter), kernel_size=k_sizes, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(int(n_filter))
        )

    def forward(self, input):
        output = self.model(input)
        return output

class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias = True, dilation = 1, is_batchnorm = True):
        super(Conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d( int(in_channels),int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation )

        if is_batchnorm:
            self.model = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.model = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

class DeConv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channel, n_filter, k_sizes, stride, padding, bias = True):
        super(DeConv2DBatchNormRelu, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(int(in_channel), int(n_filter), kernel_size=k_sizes, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(int(n_filter)),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.model(input)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, n_filter, stride = 1, downsample = None ):
        super(ResidualBlock, self).__init__()

        self.convbnrelu1 = Conv2DBatchNormRelu(in_channels=in_channel, n_filters=n_filter, k_size=3, stride= stride, padding=1, bias=True)
        self.convbn2 = Conv2DBatchNorm(in_channels = n_filter, n_filters = n_filter, k_size= 3 , stride=stride, padding=1, bias=True)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
         residual = x

         out = self.convbnrelu1(x)
         out = self.convbn2(out)

         if self.downsample is not None:
             residual = self.downsample(x)

         out += residual
         out = self.relu(out)
         return out

class ResidualBottleneck(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, stride = 1, downsample=None):
        super(ResidualBottleneck, self).__init__()

        self.convbnrelu1 = Conv2DBatchNormRelu(in_channel, middle_channel, 1, stride = stride, padding=0, bias=True)

        self.convbnrelu2 = Conv2DBatchNormRelu(middle_channel, middle_channel, 3, stride=1, padding=1, bias=True)

        self.convbnrelu3 = Conv2DBatchNorm(middle_channel, out_channel, 1, stride=1, padding=0, bias=True)

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        residual = input
        out = self.convbnrelu1(input)
        out = self.convbnrelu2(out)
        out = self.convbnrelu3(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual

        out = self.relu(out)
        return out

class ResidualBottleneckPSP(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, stride = 1, downsample=None):
        super(ResidualBottleneckPSP, self).__init__()

        self.convbnrelu1 = Conv2DBatchNormRelu(in_channel, middle_channel, 1, stride = stride, padding=0, bias=True)

        self.convbnrelu2 = Conv2DBatchNormRelu(middle_channel, middle_channel, 3, stride=1, padding=1, bias=True)

        self.convbnrelu3 = Conv2DBatchNorm(middle_channel, out_channel, 1, stride=1, padding=0, bias=True)

        self.convbn_hightway = Conv2DBatchNorm(in_channel, out_channel, 1, stride=stride, padding=0, bias=True)

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        residual = input

        if self.downsample is not None:
            residual = self.downsample(input)

        out = self.convbnrelu1(residual)
        out = self.convbnrelu2(out)
        out = self.convbnrelu3(out)


        residual = self.convbn_hightway(residual)

        out += residual

        out = self.relu(out)
        return out

class DlatedBottleneck(nn.Module):
    def __init__(self, in_channel,middle_channel, out_channel, stride = 1, dilation = 1):
        super(DlatedBottleneck, self).__init__()

        self.convbnrelu1 = Conv2DBatchNormRelu(in_channel, middle_channel, 1, stride=stride, padding=0, bias=True)

        self.convbnrelu2 = Conv2DBatchNormRelu(middle_channel, middle_channel, 1, stride=1, padding=0, bias=True, dilation=dilation)

        self.convbnrelu3 = Conv2DBatchNorm(middle_channel, out_channel, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.convbnrelu3(self.convbnrelu2(self.convbnrelu1(input)))
        output = self.relu(output)
        return output


class DlatedBottleneckPSP(nn.Module):
    def __init__(self, in_channel,middle_channel, out_channel, stride = 1, dilation = 1):
        super(DlatedBottleneckPSP, self).__init__()

        self.convbnrelu1 = Conv2DBatchNormRelu(in_channel, middle_channel, 1, stride=stride, padding=0, bias=True)

        if dilation > 1:
            self.convbnrelu2 = Conv2DBatchNormRelu(middle_channel, middle_channel, 1, stride=1, padding=0, bias=True, dilation=dilation)
        else:
            self.convbnrelu2 = Conv2DBatchNormRelu(middle_channel, middle_channel, 1, stride=1, padding=0, bias=True, dilation=1)

        self.convbnrelu3 = Conv2DBatchNorm(middle_channel, out_channel, 1, stride=1, padding=0, bias=True)

        self.convbnrelu_highway = Conv2DBatchNorm(in_channel, out_channel, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        hight_way = self.convbnrelu_highway(input)

        out = self.convbnrelu1(input)
        out = self.convbnrelu2(out)
        out = self.convbnrelu3(out)

        out += hight_way

        out = self.relu(out)

        return out

class PSPNet(nn.Module):
    def __init__(self, in_channel, middle_channel):
        super(PSPNet, self).__init__()

        self.convbnrelu1 = Conv2DBatchNormRelu(in_channel, middle_channel, 1, stride=1, padding=0, bias=True)

        self.AvgPool2d1 = nn.AvgPool2d(2)
        self.AvgPool2d2 = nn.AvgPool2d(3)
        self.AvgPool2d3 = nn.AvgPool2d(4)

    def forward(self, input):
        h,w = input.shape[2:]

        out1 = self.AvgPool2d1(input)
        out1 = F.interpolate(out1, size=(h, w), mode="bilinear")

        out2 = self.AvgPool2d2(input)
        out2 = F.interpolate(out2, size=(h, w), mode="bilinear")

        out3 = self.AvgPool2d3(input)
        out3 = F.interpolate(out3, size=(h, w), mode="bilinear")

        out = out1 + out2 + out3 + input

        out = self.convbnrelu1(out)

        return out


