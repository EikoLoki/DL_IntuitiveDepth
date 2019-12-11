# @Author:Xiran

import torch
import torch.nn as nn
import torch.nn.functional as F

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=dilation if dilation > 1 and kernel_size > 1 else pad, dilation=dilation),
        nn.BatchNorm2d(out_planes)
    )

def deconvbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation),
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
        self.conv2 = convbn(out_planes, out_planes, 3, 1, pad, dilation)

        self.downsample_feature = convbn(in_planes, out_planes, 1, stride, 0, dilation)

    def forward(self, x):
        out = self.conv1(x)
        # print('Resblock conv1 size:', out.size())
        out = self.conv2(out)
        # print('Resblock conv2 size:', out.size())
        
        if self.in_planes != self.out_planes:
            x = self.downsample_feature(x)
        
        out += x
        # print('addition success')
        out = F.relu(out, inplace=True)
        return out
