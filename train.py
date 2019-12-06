# Author: Xiran Zhang, Pengfei Li
# training are implemented here


import argparse
import torch
import torch.nn as nn

import dataloader.listfile as lf
import dataloader.loader as ld
from model import *

parser = argparse.ArgumentParser(description='SPAutoED')
parser.add_argument('--datapath', default='/media/xiran_zhang/2011_HD7/EndoVis_SCARED')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loadmodel', default='./trained_model/', help='path to the trained model, for continuous training')
parser.add_argument('--savemodel', default='./trained_model', help='folder to the save the trained model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='seed for random (default:1)')

args = parser.parse_args()

# set train environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('torch cuda status:', torch.cuda.is_available())
if torch.cuda.is_available():
    cuda = True
else
    EnvironmentError('cuda status error.')


# load all data
train_left_img, train_right_img, train_cam_para,\
    val_left_img, val_right_img, val_cam_para, \
    test_left_img, test_right_img, test_cam_para = lf.SCARED_lister(args.datapath)
    
trainImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(train_left_img, train_right_img, train_cam_para, training=True),
    batch_size=5, shuffle=True, num_workers=5, drop_last=False
    )
valImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(val_left_img, val_right_img, val_cam_para, training=False),
    batch_size=5, shuffle=False, num_workers=5, drop_last=False
)
testImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(test_left_img, test_right_img, test_cam_para, training=False),
    batch_size=5, shuffle=False, num_workers=5, drop_last=False
)

# create model
model = SPAutoED()
if cuda:
    model = nn.DataParallel()
    model.cuda()
pasued_epoch = 0
