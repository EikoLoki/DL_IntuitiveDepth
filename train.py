# Author: Xiran Zhang, Pengfei Li
# training are implemented here

import os
import time
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import dataloader.listfile as lf
import dataloader.loader as ld
from model import LossFunc
from model import SPAutoED as SPEDNet
from utils.disp_utils import depth_to_disp

parser = argparse.ArgumentParser(description='SPAutoED')
parser.add_argument('--datapath', default='/media/xiran_zhang/2011_HDD7/EndoVis_SCARED')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loadmodel', default='./trained_model/SPAutoED_1.tar', help='path to the trained model, for continuous training')
parser.add_argument('--savemodel', default='./trained_model/', help='folder to the save the trained model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='seed for random (default:1)')

args = parser.parse_args()
writer = SummaryWriter()

# set train environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('torch cuda status:', torch.cuda.is_available())
if torch.cuda.is_available():
    cuda = True
else:
    EnvironmentError('cuda status error.')


# load all data
train_left_img, train_right_img, train_cam_para,\
    val_left_img, val_right_img, val_cam_para, \
    test_left_img, test_right_img, test_cam_para = lf.SCARED_lister(args.datapath)
    
trainImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(train_left_img, train_right_img, train_cam_para, training=True),
    batch_size=1, shuffle=True, num_workers=5, drop_last=False
    )
valImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(val_left_img, val_right_img, val_cam_para, training=False),
    batch_size=1, shuffle=False, num_workers=5, drop_last=False
)
testImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(test_left_img, test_right_img, test_cam_para, training=False),
    batch_size=1, shuffle=False, num_workers=5, drop_last=False
)

# create model
model = SPEDNet.SPAutoED()
if cuda:
    model = nn.DataParallel(model)
    model = model.cuda()
pasued_epoch = 0

# load model
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    pasued_epoch = state_dict['epoch']
    train_loss = state_dict['train_loss']
    val_loss = state_dict['val_loss']

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# set criteria, optimizer and scheduler
# Option can be made here:
# optimizer: Adam, SGD
criteria = LossFunc.Loss_reonstruct()
optimizer = torch.optim.Adam(model.parameters(), lr = 10e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.7)

# train function / validation function / test function
def train(imgL, imgR, camera_para, model, optimizer):
    model.train()

    plt.imshow(imgL[0,0])
    plt.show()
    plt.imshow(imgR[0,0])
    plt.show()

    if cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    optimizer.zero_grad()
    depthL = model(imgL)
    depthR = model(imgR)
    # depth to disparity
    dispL, dispR = depth_to_disp(depthL, depthR, camera_para)
    
    # visualize
    depthL_cpu = depthL.cpu()
    depthR_cpu = depthR.cpu()
    depthL_cpu.detach_()
    depthR_cpu.detach_()
    plt.imshow(depthL_cpu[0,0])
    plt.show()
    plt.imshow(depthR_cpu[0,0])
    plt.show()


    right_l1, left_l1, lr_l1 = criteria(imgL, imgR, dispL, dispR)
    loss = 0.5*right_l1 + 0.5*left_l1 + 1.0 * lr_l1 
    # torch.autograd.set_detect_anomaly(True)
    loss.backward()
    optimizer.step()
    
    return loss.data.item()

def val(imgL, imgR, camera_para, model, optimizer, epoch):
    model.eval()
    if cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    with torch.no_grad():
        depthL = model(imgL)
        depthR = model(imgR)
        
        # visualization
        # fig, (ax1, ax2) = plt.subplots(1,2)
        # ax1.imshow(depthL)
        # ax2.imshow(dephtR)
        # plt.show()
        dispL, dispR = depth_to_disp(depthL, depthR, camera_para)

        right_l1, left_l1, lr_l1 = criteria(imgL, imgR, dispL, dispR)
        loss = 0.5*right_l1 + 0.5*left_l1 + 1.0 * lr_l1

    return loss.data.item()


def main():
    start_epoch = pasued_epoch
    total_epochs = args.epochs
    print('Start training ...')

    total_iteration = 0
    for epoch in range(start_epoch, total_epochs):
        print('\nEPOCH ' + str(epoch + 1) + ' of ' + str(total_epochs) + '\n')
        #-----------------Training---------------
        train_batch_num = 0
        epoch_train_loss = 0
        
        for idx, (left, right, para_dict) in enumerate(tqdm(trainImgLoader)):
            start_time = time.time()
            train_loss = train(left, right, para_dict, model, optimizer)
            epoch_train_loss += train_loss
            train_batch_num += 1
            total_iteration += 1
            print('Iter %d training loss = %.3f, time =  %.2f' %(idx, train_loss, time.time() - start_time))
            writer.add_scalar('Iteration loss', train_loss, total_iteration+1)

        epoch_train_loss /= train_batch_num
        writer.add_scalar('Loss/train', epoch_train_loss, epoch+1)


        #----------------Validation--------------
        val_batch_num = 0
        epoch_val_loss = 0
        for idx, (left, right, para_dict) in enumerate(tqdm(valImgLoader)):
            start_time = time.time()
            val_loss = val(left, right, para_dict, model, optimizer, epoch)
            epoch_val_loss += val_loss
            val_batch_num += 1
            print('Iter %d val loss = %.3f, time =  %.2f' %(idx, val_loss, time.time() - start_time))

        epoch_val_loss /= val_batch_num
        writer.add_scalar('Loss/val', epoch_val_loss, epoch+1)

        savefilename = args.savemodel + 'SPAutoED_' + str(epoch+1) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss':epoch_train_loss,
            'val_loss':epoch_val_loss,
        }, savefilename)

    writer.close()

if __name__ == '__main__':
    main()
    # test()