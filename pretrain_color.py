# Author: Xiran Zhang, Pengfei Li
# training are implemented here

import os
import time
import argparse
import sys
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
from utils.disp_utils import depth_to_disp,visualize

parser = argparse.ArgumentParser(description='SPAutoED')
parser.add_argument('--datapath', default='data/EndoVis_SCARED_subset')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--loadmodel', default='trained_model/coloization/colorization_v3.tar', help='path to the trained model, for continuous training')
parser.add_argument('--savemodel', default='./trained_model/', help='folder to the save the trained model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='seed for random (default:1)')
parser.add_argument('--use_cuda', type=bool, default=True, help='if true, use gpu')
args = parser.parse_args()
writer = SummaryWriter()

# set train environment
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('torch cuda status:', torch.cuda.is_available())
if torch.cuda.is_available():
    cuda = True
else:
    EnvironmentError('cuda status error.')
DEBUG = True

# load all data
train_left_img, train_right_img, train_cam_para,\
    val_left_img, val_right_img, val_cam_para, \
    test_left_img, test_right_img, test_cam_para = lf.SCARED_lister(args.datapath)
    
trainImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(train_left_img, train_right_img, train_cam_para, training=True),
    batch_size=60, shuffle=True, num_workers=8, drop_last=False
    )
valImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(val_left_img, val_right_img, val_cam_para, training=False),
    batch_size=60, shuffle=False, num_workers=8, drop_last=False
)
testImgLoader = torch.utils.data.DataLoader(
    ld.SCARED_loader(test_left_img, test_right_img, test_cam_para, training=False),
    batch_size=70, shuffle=False, num_workers=8, drop_last=False
)

# create model
model = SPEDNet.SPAutoED_colorization()
model = nn.DataParallel(model)

# load model
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel, map_location={'cuda:0': 'cpu'})
    for k in state_dict['state_dict']:
        if "module.SP_extractor.conv0" in k:
            print(state_dict['state_dict'][k])
    sys.exit()
    model.load_state_dict(state_dict['state_dict'])
    pasued_epoch = state_dict['epoch']
    train_loss = state_dict['train_loss']
    val_loss = state_dict['val_loss']
    scheduler = state_dict['scheduler']
    optimizer = state_dict['optimizer']
    
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



# set criteria, optimizer and scheduler
# Option can be made here:
# optimizer: Adam, SGD

optimizer = torch.optim.Adam(model.parameters(), lr = 2*10e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.7)
if cuda and args.use_cuda:
    model = model.cuda()
else:
    model.to("cpu")

# train function / validation function / test function
def pretrain(imgL, imgR, model, criterion, optimizer):
    model.train()
    imgL_grey = (0.299*imgL[:,0,:,:] + 0.587*imgL[:,1,:,:] + 0.114*imgL[:,2,:,:]).unsqueeze(1)
    imgR_grey = (0.299*imgR[:,0,:,:] + 0.587*imgR[:,1,:,:] + 0.114*imgR[:,2,:,:]).unsqueeze(1)
    if cuda:
        imgL_grey, imgR_grey = imgL_grey.cuda(), imgR_grey.cuda()
        imgL, imgR = imgL.cuda(), imgR.cuda()
    
    optimizer.zero_grad()

    img_color_L = model(imgL_grey)
    img_color_R = model(imgR_grey)
    
    # depth to disparity
    loss_L = criterion(img_color_L, imgL)
    loss_R = criterion(img_color_R, imgR)
    loss = loss_L +loss_R
    loss.backward()
    optimizer.step()
    if DEBUG:
        # visualization
        left_recons_cpu = img_color_L.cpu().detach_()
        right_recons_cpu = img_color_R.cpu().detach_()
        left_data_cpu = imgL.cpu().detach_()
        right_data_cpu = imgR.cpu().detach_()
        
        for i in range(left_data_cpu.shape[0]):
            data_path = "temp_vis/"
            visualize(left_data_cpu[i,:,:,:], right_data_cpu[i,:,:,:], left_recons_cpu[i,:,:], right_recons_cpu[i], save_path=data_path+"left_{}.jpg".format(i))
        print(loss.data.item())
        import sys
        sys.exit()
    
    return loss.data.item()

def val_color(imgL, imgR, model, criterion, optimizer):
    model.eval()
    if cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()


    with torch.no_grad():
        imgL_grey = (0.299*imgL[:,0,:,:] + 0.587*imgL[:,1,:,:] + 0.114*imgL[:,2,:,:]).unsqueeze(1)
        imgR_grey = (0.299*imgR[:,0,:,:] + 0.587*imgR[:,1,:,:] + 0.114*imgR[:,2,:,:]).unsqueeze(1)
        if cuda:
            imgL_grey, imgR_grey = imgL_grey.cuda(), imgR_grey.cuda()
            imgL, imgR = imgL.cuda(), imgR.cuda()
        
        optimizer.zero_grad()

        img_color_L = model(imgL_grey)
        img_color_R = model(imgR_grey)
        
        # depth to disparity
        loss_L = criterion(img_color_L, imgL)
        loss_R = criterion(img_color_R, imgR)
        loss = loss_L +loss_R
        if DEBUG:
            # visualization
            left_recons_cpu = img_color_L.cpu().detach_()
            right_recons_cpu = img_color_R.cpu().detach_()
            left_data_cpu = imgL.cpu().detach_()
            right_data_cpu = imgR.cpu().detach_()
            
            for i in range(left_data_cpu.shape[0]):
                data_path = "temp_vis/"
                visualize(left_data_cpu[i,:,:,:], right_data_cpu[i,:,:,:], left_recons_cpu[i,:,:], right_recons_cpu[i], save_path=data_path+"left_{}.jpg".format(i))
            print(loss.data.item())
            sys.exit()

    return loss.data.item() 


def main():
    start_epoch = pasued_epoch+1
    total_epochs = args.epochs
    print('Start training ...')

    total_iteration = 0
    for epoch in range(0, total_epochs):
        print('\nEPOCH ' + str(epoch + 1) + ' of ' + str(total_epochs) + '\n')
        #-----------------Training---------------
        train_batch_num = 0
        epoch_train_loss = 0
        criterion = nn.MSELoss()
        for idx, (left, right, para_dict) in enumerate(tqdm(trainImgLoader)):
            start_time = time.time()
            train_loss = pretrain(left, right, model, criterion, optimizer)
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
            val_loss = val_color(left, right, model, criterion, optimizer)
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
            'optimizer':optimizer,
            'scheduler':scheduler
        }, savefilename)

        scheduler.step()
        lr = scheduler.get_lr()

    writer.close()

def test():
    criterion = nn.MSELoss()
    for idx, (left, right, para_dict) in enumerate(tqdm(valImgLoader)):
        val_loss = val_color(left, right, model, criterion, optimizer)
if __name__ == '__main__':
    #main()
    test()