# Author: Xiran Zhang, Pengfei Li
# training are implemented here


import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

import dataloader.listfile as lf
import dataloader.loader as ld
from model import LossFunc
from model import *
from utils import disp_utils

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

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    pasued_epoch = state_dict['epoch']
    max_epo_load = state_dict['max_epo']
    max_acc_load = state_dict['max_acc']

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# set criteria, optimizer and scheduler
# Option can be made here:
# optimizer: Adam, SGD
criteria = model.Loss_reconstruct()
optimizer = torch.optim.Adam(model.parameters(), lr = 10e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.7)

# train function / validation function / test function
def train(imgL, imgR, camera_para, model, optimizer):
    model.train()
    if cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    optimizer.zero_grad()

    depthL = model(imgL)
    depthR = model(imgR)

    # depth to disparity
    dispL, dispR = depth_to_disp(depthL, depthR, camera_para)

    right_l1, left_l1, lr_l1 = criteria(imgL, imgR, dispL, dispR)

    loss = 0.5*right_l1 + 0.5*left_l1 + 1.0 * lr_l1

    loss.backward()
    optimizer.step()


    