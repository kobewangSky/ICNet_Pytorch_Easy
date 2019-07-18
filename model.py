import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
import torchvision.models as models
import os
import time
from Tool import get_interp_size
from Model_Tool import (
    Conv2DBatchNorm,
DeConv2DBatchNorm,
Conv2DBatchNormRelu,
DeConv2DBatchNormRelu,
ResidualBlock,
ResidualBottleneck,
ResidualBottleneckPSP,
DlatedBottleneckPSP,
DlatedBottleneck,
PSPNet,
)

class ICnet_model(nn.Module):
    def __init__(self, n_classes = 19, input_size = (1025, 2049)):
        super(ICnet_model, self).__init__()

        self.LabelWeight = {'labe4': 0, 'labe8': 0, 'labe16': 0}

        self.num_classes = n_classes

        self.conv2DbnRelu1 = Conv2DBatchNormRelu(in_channels = 3, n_filters = 32,k_size = 3,stride = 2,padding = 1)
        self.conv2DbnRelu2 = Conv2DBatchNormRelu(in_channels=32, n_filters=32, k_size=3, stride=1, padding=1)
        self.conv2DbnRelu3 = Conv2DBatchNormRelu(in_channels=32, n_filters=64, k_size=3, stride=1, padding=1)

        self.ResidualBottleneckPSP1 = ResidualBottleneckPSP(in_channel=64, middle_channel=32, out_channel=128, downsample= nn.MaxPool2d(2) )

        self.ResidualBottleneck1 = ResidualBottleneck(in_channel=128, middle_channel=32, out_channel=128)
        self.ResidualBottleneck2 = ResidualBottleneck(in_channel=128, middle_channel=32, out_channel=128)

        self.ResidualBottleneckPSP2 = ResidualBottleneckPSP(in_channel=128, middle_channel=64, out_channel=256, stride=2)

        self.ResidualBottleneck3 = ResidualBottleneck(in_channel=256, middle_channel=64, out_channel=256)
        self.ResidualBottleneck4 = ResidualBottleneck(in_channel=256, middle_channel=64, out_channel=256)
        self.ResidualBottleneck5 = ResidualBottleneck(in_channel=256, middle_channel=64, out_channel=256)

        self.DlatedBottleneckPSP1 = DlatedBottleneckPSP(in_channel=256, middle_channel= 128, out_channel= 512, dilation=2 )

        self.DlatedBottleneck1 = DlatedBottleneck(in_channel=512, middle_channel=128, out_channel=512, dilation=2)
        self.DlatedBottleneck2 = DlatedBottleneck(in_channel=512, middle_channel=128, out_channel=512, dilation=2)
        self.DlatedBottleneck3 = DlatedBottleneck(in_channel=512, middle_channel=128, out_channel=512, dilation=2)
        self.DlatedBottleneck4 = DlatedBottleneck(in_channel=512, middle_channel=128, out_channel=512, dilation=2)
        self.DlatedBottleneck5 = DlatedBottleneck(in_channel=512, middle_channel=128, out_channel=512, dilation=2)

        self.DlatedBottleneckPSP2 = DlatedBottleneckPSP(in_channel=512, middle_channel=256, out_channel=1024, dilation=4)

        self.DlatedBottleneck6 = DlatedBottleneck(in_channel=1024, middle_channel=256, out_channel=1024, dilation=4)
        self.DlatedBottleneck7 = DlatedBottleneck(in_channel=1024, middle_channel=256, out_channel=1024, dilation=4)

        self.PSPNet1 = PSPNet(1024, 256)

        self.conv2DbnRelu_out4 = Conv2DBatchNormRelu(in_channels=256, n_filters=128,k_size=3,  stride=1, padding=2, dilation=2)

        self.conv2Dbn1 = Conv2DBatchNorm(in_channels=256, n_filters=128, k_size=1, stride=1, padding=0)
        self.Dilaconv2Dbn2 = Conv2DBatchNorm(in_channels=128, n_filters=128, k_size=3, stride=1, padding=2, dilation=2)


        self.conv2DbnRelu1_ori_size = Conv2DBatchNormRelu( in_channels=3, n_filters=32, k_size=3, stride=2, padding=1)
        self.conv2DbnRelu2_ori_size = Conv2DBatchNormRelu(in_channels=32, n_filters=32, k_size=3, stride=2, padding=1)
        self.conv2DbnRelu3_ori_size = Conv2DBatchNormRelu(in_channels=32, n_filters=64, k_size=3, stride=2, padding=1)
        self.conv2Dbn1_ori_size = Conv2DBatchNorm(in_channels=64, n_filters=128, k_size=3, stride=1, padding=1)


        self.output4 = nn.Conv2d( in_channels=128, out_channels=n_classes, kernel_size=1, padding=0, stride=1)
        self.output8 = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1, padding=0, stride=1)
        self.output16 = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1, padding=0, stride=1)


        self.logits4 = None
        self.logits8 = None
        self.logits16 = None

    def forward(self, input):


        out_size_2 = F.interpolate(input, scale_factor= 0.5)

        out_size_4 = self.conv2DbnRelu1(out_size_2)
        out = self.conv2DbnRelu2(out_size_4)
        out = self.conv2DbnRelu3(out)

        out_size_8 = self.ResidualBottleneckPSP1(out)

        out = self.ResidualBottleneck1(out_size_8)
        out = self.ResidualBottleneck2(out)

        out_size_16 = self.ResidualBottleneckPSP2(out)

        out_size_32 = F.interpolate(out_size_16, scale_factor=0.5)

        out = self.ResidualBottleneck3(out_size_32)
        out = self.ResidualBottleneck4(out)
        out = self.ResidualBottleneck5(out)

        out = self.DlatedBottleneckPSP1(out)

        out = self.DlatedBottleneck1(out)
        out = self.DlatedBottleneck2(out)
        out = self.DlatedBottleneck3(out)
        out = self.DlatedBottleneck4(out)
        out = self.DlatedBottleneck5(out)

        out = self.DlatedBottleneckPSP2(out)

        out = self.DlatedBottleneck6(out)
        out = self.DlatedBottleneck7(out)

        out = self.PSPNet1(out)

        out = F.interpolate(out, scale_factor=2)
        out_deeper_16 = self.conv2DbnRelu_out4(out)

        out2_size16 = self.conv2Dbn1(out_size_16)

        out2_size16 = out_deeper_16 + out2_size16

        out2_size16 = F.relu(out2_size16)

        out2_size8 = F.interpolate(out2_size16, scale_factor=2)

        out2_size4 = self.Dilaconv2Dbn2(out2_size8)

        out = self.conv2DbnRelu1_ori_size(input)
        out = self.conv2DbnRelu2_ori_size(out)
        out = self.conv2DbnRelu3_ori_size(out)
        out = self.conv2Dbn1_ori_size(out)

        out = out + out2_size4

        out2_size4 = F.interpolate(out, scale_factor=2)

        out2_size4 = self.output4(out2_size4)

        out2_size16 = self.output16(out2_size16)

        out2_size8 = self.output8(out2_size8)

        if self.training:
            return out2_size4, out2_size8, out2_size16
        else:
            out2_size4 = F.interpolate(
                out2_size4,
                size=get_interp_size(out2_size4, z_factor=4),
                mode="bilinear",
                align_corners=True,
            )
            return out2_size4




    def loss(self, logits4 : Tensor, logits8 : Tensor, logits16 : Tensor, labels):


        out4 = self.cross_entropy2d(logits4, labels)
        out8 = self.cross_entropy2d(logits8, labels)
        out16 = self.cross_entropy2d(logits16, labels)

        out = (out4 * self.LabelWeight['label4']) + (out8 * self.LabelWeight['label8']) + (out16 * self.LabelWeight['label16'])
        return out

    def save(self, path : str, name : str):
        #path = os.path.join(path, 'mode-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))

        path = os.path.join(path, name) + '.pth'

        torch.save(self.state_dict(), path)

        return path

    def load(self, path:str):
        self.load_state_dict(torch.load(path))
        return self

    def cross_entropy2d(self, input, target):
        n,c,h,w = input.size()
        nt, ht, wt = target.size()

        if h != ht:
            input = F.interpolate(input, (ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1).reshape(-1, 1)

        loss = F.cross_entropy(
            input, target, ignore_index=250
        )


        return loss

    def setuplabelweight(self, lableweight):
        self.LabelWeight = lableweight


