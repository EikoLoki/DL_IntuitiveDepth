# @Author: Xiran Zhang
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodule import *

class SPAutoED(nn.Module):
    def __init__(self):
        super(SPAutoED, self).__init__()
        self.SP_extractor = SP_Feature_Extractor()
        self.EDcoder = AutoED()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # extract features:
        feature = self.SP_extractor(x)
        depth = self.EDcoder(feature,x)
        return depth


class SP_Feature_Extractor(nn.Module):
    def __init__(self):
        super(SP_Feature_Extractor, self).__init__()
        
        # feature extractor part:
        self.conv0 = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True)
        )                                       # output size: H/2 * W/2 * 32

        self.layer1 = self._make_layer(ResBlock, 32, 32, 3, 2, 1, 1)    # output size: H/4*W/4*32  
        self.layer2 = self._make_layer(ResBlock, 32, 64, 3, 1, 1, 1)   # output size: H/4*W/4*64
        self.layer3 = self._make_layer(ResBlock, 64, 128, 3, 1, 1, 2)   # output size: H/4*W/4*128 dila = 2
        self.layer4 = self._make_layer(ResBlock, 128, 128, 3, 1, 1, 4)  # output size: H/4*W/4*128 dila = 4

        # SPP part:
        self.branch1 = nn.Sequential(
            nn.AvgPool2d((32,32), stride=(32,32)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((16,16), stride=(16,16)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((8,8), stride=(8,8)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((4,4), stride=(4,4)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        # merge part:
        self.fusion = nn.Sequential(
            convbn(320,128,3,1,1,1),
            nn.ReLU(inplace=True),  # output size: H/4*W/4*128
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True),  # output size: H/4*W/4*32 
            )

    def _make_layer(self, block, in_planes, out_planes, block_num, stride, pad, dilation):
        layers = []
        layers.append(block(in_planes, out_planes, stride, pad, dilation))
        for i in range(1, block_num):
            layers.append(block(out_planes, out_planes, 1, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv0(x)
        # print('conv0 size:', output.size())
        output = self.layer1(output)
        # print('layer1 output size:', output.size())
        output_raw = self.layer2(output)
        # print('layer2 output size:', output_raw.size())
        output = self.layer3(output_raw)
        # print('layer3 output size:', output.size())
        output_dila = self.layer4(output)
        # print('layer4 output size:', output_dila.size())

        output_branch1 = self.branch1(output_dila)
        output_branch1 = F.interpolate(output_branch1, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_dila)
        output_branch2 = F.interpolate(output_branch2, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_dila)
        output_branch3 = F.interpolate(output_branch3, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_dila)
        output_branch4 = F.interpolate(output_branch4, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_dila, output_branch1, output_branch2, output_branch3, output_branch4), 1)
        output_feature = self.fusion(output_feature)

        return output_feature 


class AutoED(nn.Module):
    def __init__(self):
        super(AutoED, self).__init__()
        # This AutoED does not have resblock inside, but has skip connection.
        # input size: 1/4H*1/4*W*32
        # Encoder part:
        self.conv1 = nn.Sequential(
            convbn(32, 64, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/4H*1/4*W*64
            convbn(64, 64, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/4H*1/4*W*64
                           
        )

        self.conv2 = nn.Sequential(
            # nn.MaxPool2d(2,2),              # output size: 1/8H*1/8W*64
            convbn(64, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/8H*1/8W*128
            convbn(128, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/8H*1/8W*128                 
        )

        self.conv3 = nn.Sequential(
            # nn.MaxPool2d(2,2),              # output size: 1/16H*1/16W*128
            convbn(128, 256, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/16H*1/16W*256
            convbn(256, 256, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/16H*1/16W*256
            convbn(256, 256, 3, 1, 1, 1),
            nn.ReLU(inplace=True)           # output size: 1/16H*1/16W*256
        )

        self.conv4 = nn.Sequential(
            # nn.MaxPool2d(2,2),              # output size: 1/32H*1/32W*256
            convbn(256, 512, 3, 1, 1, 1), 
            nn.ReLU(inplace=True),          # output size: 1/32H*1/32W*512
            convbn(512, 512, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/32H*1/32W*512
            convbn(512, 512, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/32H*1/32W*512

        )

        # Connection part:
        self.coder = nn.Sequential(
            # nn.MaxPool2d(2,2),              # output size: 1/64H*1/64W*512
            convbn(512, 512, 3, 2, 1, 1),   
            nn.ReLU(inplace=True),          # output size: 1/64H*1/64W*512
            deconvbn(512, 512, 3, 2, 1, 1, 1)
            # nn.MaxUnpool2d(2,2)             # output size: 1/32H*1/32W*512
        )

        # Decoder part:
        self.deconv1 = nn.Sequential(
            deconvbn(512, 512, 3, 2, 1, 1, 1), 
            nn.ReLU(inplace=True),          # output size: 1/32H*1/32W*512
            convbn(512, 512, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # output size: 1/32H*1/32W*512
            convbn(512, 256, 3, 1, 1, 1),
            nn.ReLU(inplace=True)           # output size: 1/32H*1/32W*256
        )

        self.deconv2 = nn.Sequential(
            # nn.MaxUnpool2d(2,2),            # output size: 1/16H*1/16W*256
            deconvbn(512, 256, 3, 2, 1, 1, 1),     # input change to 512
            nn.ReLU(inplace=True),          # output size: 1/16H*1/16W*256
            convbn(256, 256, 3, 1, 1, 1),
            nn.ReLU(inplace=True),          # ouptut size: 1/16H*1/16W*256
            convbn(256, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True)           # output size: 1/16H*1/16W*128
        )

        self.deconv3 = nn.Sequential(
            # nn.MaxUnpool2d(2,2),            # output size: 1/8H*1/8W*128
            deconvbn(256, 128, 3, 2, 1, 1, 1),     # input change to 256
            nn.ReLU(inplace=True),          # output size: 1/8H*1/8W*128
            convbn(128, 64, 3, 1, 1, 1),  
            nn.ReLU(inplace=True)           # output size: 1/8H*1/8W*64
        )

        self.deconv4 = nn.Sequential(
            # nn.MaxUnpool2d(2,2),            # output size: 1/4H*1/4W*64
            deconvbn(128, 64, 3, 2, 1, 1, 1),      # input change to 128
            nn.ReLU(inplace=True),          # output size: 1/4H*1/4W*64
            convbn(64, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True)           # output size: 1/4H*1/4W*32
        )

        self.pool = nn.MaxPool2d(2,2)
        self.unpool = nn.MaxUnpool2d(2,2)

        # Upsampling part:
        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResBlock(32, 16, 1, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.refine = nn.Sequential(
            ResBlock(19, 1, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, img):
        # input size: 1/4H*1/4*W*32
        # Encoder part:
        # print('input size:', x.size())
        out1 = self.pool(self.conv1(x))            # output: 1/4(HxW)x64, skip connect to entry of deconv4
        out2 = self.pool(self.conv2(out1))         # output: 1/8(HxW)x128, skip connect to entry of deconv3
        out3 = self.pool(self.conv3(out2))         # output: 1/16(HxW)x256, skip connect to entry of deconv2
        out4 = self.pool(self.conv4(out3))                              # output: 1/32(HxW)x512, skip connect to entry of deconv1
        # connection part:
        # print('out size:', out4.size())
        code = self.coder(out4)
        # code = F.interpolate(code, scale_factor=2)               # output: 1/32(HxW)x512

        # Decoder part:
        code = code + out4
        # print('code size:', code.size())
        input2 = self.deconv1(code)     # output: 1/32(HxW)x256

        # input2 = F.interpolate(input2, scale_factor=2)
        # out3 = F.interpolate(out3, scale_factor=2)
        input2 = torch.cat([input2,out3],1)
        input3 = self.deconv2(input2)   # output: 1/16(HxW)x128

        # input3 = F.interpolate(input3,scale_factor=2)
        # out2 = F.interpolate(out2, scale_factor=2)
        input3 = torch.cat([input3,out2],1)
        input4 = self.deconv3(input3)   # output: 1/8(HxW)x64

        # input4 = F.interpolate(input4, scale_factor=2)
        # out1 = F.interpolate(out1, scale_factor=2)
        input4 = torch.cat([input4,out1],1)
        result = self.deconv4(input4)   # output: 1/4(HxW)x32
        # upsample part:
        
        # print('result size:', result.size())
        depth = self.upsample(result)
        # print('depth size:', depth.size())
        # print('img size:', img.size())
        # depth = self.refine(depth)
        depth = torch.cat([depth, img], 1)
        depth = 0.2*960.0*self.refine(depth)
        return depth