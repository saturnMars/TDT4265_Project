import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import math
from .attention_blocks import DualAttBlock
from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc
import cv2
from .norm import Norm2d


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
            )


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(CenterBlock, self).__init__()
        self.in_channels = in_channels
        self.is_deconv=is_deconv

        if self.is_deconv: #Dense Block
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels)
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, middle_channels)
            self.convUp = nn.Sequential(
                                nn.ConvTranspose2d(in_channels+2*middle_channels, out_channels,
                                                kernel_size=4, stride=2, padding=1),
                                nn.ReLU(inplace=True)
            )

        else:
            self.convUp = nn.Unsample(scale_factor=2, mode='bilinear', align_corners=True),
            self.conv1 = conv3x3_bn_relu(in_channels, middle_channels),
            self.conv2 = conv3x3_bn_relu(in_channels+middle_channels, out_channels)

    def forward(self, x):
        tmp = []
        if self.is_deconv == False:
            convUp = self.convUp(x); tmp.append(convUp)
            conv1 = self.conv1(convUp); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1))
            return conv2

        else:
            tmp.append(x)
            conv1 = self.conv1(x); tmp.append(conv1)
            conv2 = self.conv2(torch.cat(tmp, 1)); tmp.append(conv2)
            convUp = self.convUp(torch.cat(tmp, 1))
            return convUp


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                conv3x3_bn_relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv3x3_bn_relu(in_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

        ## initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class SAUNet(nn.Module): #SAUNet
    def __init__(self, num_classes=4, num_filters=32, pretrained=True, is_deconv=True):
        super(SAUNet, self).__init__()

        self.num_classes = num_classes
        print("SAUNet w/ Shape Stream")
        self.pool = nn.MaxPool2d(2,2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)

        self.shape_inp = nn.Conv2d(1, 3, kernel_size=1, padding=0)
        
        #for n, p in self.encoder.named_parameters():
        #    print(n)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        #Shape Stream
        self.c3 = nn.Conv2d(256, 1, kernel_size=1)
        self.c4 = nn.Conv2d(512, 1, kernel_size=1)
        self.c5 = nn.Conv2d(1024, 1, kernel_size=1)

        self.d0 = nn.Conv2d(128, 64, kernel_size=1)
        self.res1 = ResBlock(64, 64)
        self.d1 = nn.Conv2d(64, 32, kernel_size=1)
        self.res2 = ResBlock(32, 32)
        self.d2 = nn.Conv2d(32, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)

        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))

        #Encoder
        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        #Decoder
        self.center = conv3x3_bn_relu(1024, num_filters * 8 * 2)
        self.dec5 = DualAttBlock(inchannels=[512, 1024], outchannels=512)
        self.dec4 = DualAttBlock(inchannels=[512, 512], outchannels=256)
        self.dec3 = DualAttBlock(inchannels=[256, 256], outchannels=128)
        self.dec2 = DualAttBlock(inchannels=[128,  128], outchannels=64)
        self.dec1 = DecoderBlock(64, 48, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters*2, num_filters)

        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        
        x = self.shape_inp(x)
        x_size = x.size()
        
        #Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2t(self.conv2(conv1))
        conv3 = self.conv3t(self.conv3(conv2))
        conv4 = self.conv4t(self.conv4(conv3))
        conv5 = self.conv5(conv4)

        #Shape Stream
        ss = F.interpolate(self.d0(conv2), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(conv3), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss = self.gate1(ss, c3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(conv4), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.gate2(ss, c4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(conv5), x_size[2:],
                            mode='bilinear', align_corners=True)
        ss = self.gate3(ss, c5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)

        ### Canny Edge
        im_arr = np.mean(x.cpu().detach().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        ### End Canny Edge

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        edge = self.expand(acts)

        #Decoder
        conv2 = F.interpolate(conv2, scale_factor=2, mode='bilinear', align_corners=True)
        conv3 = F.interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        conv4 = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)

        center = self.center(self.pool(conv5))
        dec5, _ = self.dec5([center, conv5])
        dec4, _ = self.dec4([dec5, conv4])
        dec3, att = self.dec3([dec4, conv3])
        dec2, _ = self.dec2([dec3, conv2])
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(torch.cat([dec1, edge], dim=1))

        x_out = self.final(dec0)

        att = F.interpolate(att, scale_factor=4, mode='bilinear', align_corners=True)

        return nn.functional.softmax(x_out, dim=1) #, edge_out, att

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                        diffY // 2, diffY - diffY //2))
