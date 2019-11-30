# @Author: Xiran Zhang
import torch
import torch.nn as nn
import torch.nn.functional as F
import submodule

class SPAutoED(nn.Module):
    def __init__(self):
        super(SPAutoED, self).__init__()
        



class SPEncoder(nn.Module):
    def __init__(self):
        super(SPEncoder, self).__init__()
        
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
        self.layer2 = self._make_layer(ResBlock, 32, 64, 16, 1, 1, 1)   # output size: H/4*W/4*64
        self.layer3 = self._make_layer(ResBlock, 64, 128, 3, 1, 1, 2)   # output size: H/4*W/4*128 dila = 2
        self.layer4 = self._make_layer(ResBlock, 128, 128, 3, 1, 1, 4)  # output size: H/4*W/4*128 dila = 4

        # SPP part:
        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64,64), stride=(64,64)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32,32), stride=(32,32)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16,16), stride=(16,16)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8,8), stride=(8,8)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )

        # merge part:
        self.encode = nn.Sequential(
            convbn(320,128,3,2,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(),             # output size: H/8*W/8*128
            convbn(128,256,3,2,1,1),
            nn.ReLU(inplace=True),      
            nn.MaxPool2d(),             # output size: H/16*W/16*256
            convbn(256,512,3,1,1,1),     
            nn.ReLU(inplace=True),
            nn.MaxPool2d()              # output size: H/32*H/32*512
            )

    def _make_layer(self, block, in_planes, out_planes, block_num, stride, pad, dilation):
        layers = []
        layers.append(block(in_planes, out_planes, stride, pad, dilation))
        for i in range(1, block_num):
            layers.append(block(in_planes, out_planes, 1, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv0(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_dila = self.layer4(output)

        output_branch1 = self.branch1(output_dila)
        output_branch1 = F.upsample(output_branch1, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_dila)
        output_branch2 = F.upsample(output_branch2, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_dila)
        output_branch3 = F.upsample(output_branch3, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_dila)
        output_branch4 = F.upsample(output_branch4, (output_dila.size()[2],output_dila.size()[3]),mode='bilinear')


        output_feature = torch.cat((output_raw, output_dila, output_branch1, output_branch2, output_branch3, output_branch4), 1)
        output_feature = self.encode(output_feature)

        return output_feature 
