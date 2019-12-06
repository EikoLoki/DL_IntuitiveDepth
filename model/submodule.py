# @Author:Xiran

import torch
import torch.nn as nn

def convbn(in_planes, out_planes, kernel_size, stride, pad=0, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation),
        nn.BatchNorm2d(out_planes)
    )

def deconvbn(in_planes, out_planes, kernel_size, stride, pad=0, dilation):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation),
        nn.BatchNorm2d(out_planes)
    )


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, pad, dilation):
        super(ResBlock, self).__init__()

        self.in_planes = in_planes,
        self.out_planes = out_planes
        self.conv1 = nn.Sequential(
            convbn(in_planes, out_planes, 3, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )
        self.conv2 = convbn(out_planes, out_planes, 3, stride, pad, dilation)

    def downsample_feature(self):
        return convbn(in_planes, out_planes, 1, stride=self.stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.covn2(out)

        if self.in_planes != self.out_planes:
            x = self.downsample_feature(x)
        
        out += x
        out = F.relu(out, inplace=True)
        return out
