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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataloader.sf_listfile as lf
import dataloader.sf_loader as ld
from model import monodepth_loss
from model import resnet18_md as Res18_md
from utils.disp_utils import depth_to_disp, visualize6

parser = argparse.ArgumentParser(description='SPAutoED')
parser.add_argument('--datapath', default='/media/xiran_zhang/Crypto/sceneflow/')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--loadmodel', default='./trained_model/Res18_md_1.tar', help='path to the trained model, for continuous training')
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
all_left_img, all_right_img, test_left_img, test_right_img = lf.dataloader(args.datapath)

trainImgLoader = torch.utils.data.DataLoader(
         ld.myImageFloder(all_left_img,all_right_img, True), 
         batch_size= 5, shuffle= True, num_workers= 5, drop_last=False)

valImgLoader = torch.utils.data.DataLoader(
         ld.myImageFloder(test_left_img,test_right_img, False), 
         batch_size= 15, shuffle= False, num_workers= 5, drop_last=False)

# create model
model = Res18_md.Resnet18_md(6)
if cuda:
    model = nn.DataParallel(model)
    model = model.cuda()
pasued_epoch = -1


# set criteria, optimizer and scheduler
# Option can be made here:
# optimizer: Adam, SGD
criteria = monodepth_loss.MonodepthLoss(4, 0.85, 0.1, 1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.7)

# load model
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    pasued_epoch = state_dict['epoch']
    train_loss = state_dict['train_loss']
    # val_loss = state_dict['val_loss']
    scheduler = state_dict['scheduler']
    optimizer = state_dict['optimizer']
    # print('train loss:', train_loss, 'val loss:', val_loss)
    

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# train function / validation function / test function
def train(imgL, imgR, model, optimizer):
    model.train()

    # plt.imshow(imgL[0,0])
    # plt.show()
    # plt.imshow(imgR[0,0])
    # plt.show()
    _,_,h,w = imgL.size()
    imgL1, imgR1 = imgL, imgR
    # imgL2, imgR2 = F.interpolate(imgL, size=[h//2,w//2], mode='bilinear', align_corners=True), F.interpolate(imgR, size=[h//2,w//2], mode='bilinear', align_corners=True)
    # imgL3, imgR3 = F.interpolate(imgL, size=[h//4,w//4], mode='bilinear', align_corners=True), F.interpolate(imgR, size=[h//4,w//4], mode='bilinear', align_corners=True)
    # imgL4, imgR4 = F.interpolate(imgL, size=[h//8,w//8], mode='bilinear', align_corners=True), F.interpolate(imgR, size=[h//8,w//8], mode='bilinear', align_corners=True)
    
    
    if cuda:
        imgL1, imgR1 = imgL1.cuda(), imgR1.cuda()
        # imgL2, imgR2 = imgL2.cuda(), imgR2.cuda()
        # imgL3, imgR3 = imgL3.cuda(), imgR3.cuda()
        # imgL4, imgR4 = imgL4.cuda(), imgR4.cuda()

    optimizer.zero_grad()
    x = torch.cat([imgL1, imgR1], 1)
    disps = model(x)

    # disps = []
    # for i in range(len(dispsL)):
    #     disps.append(torch.cat([dispsL[i], dispsR[i]], 1))

    loss = criteria(disps,[imgL1, imgR1])
    disp1, disp2, disp3, disp4 = disps
    disp1 = disp1.cpu().detach_()
    disp2 = disp2.cpu().detach_()
    disp3 = disp3.cpu().detach_()
    disp4 = disp4.cpu().detach_()
    
    visualize6(imgL[0,0], imgR[0,0], disp1[0,0], disp2[0,0], disp3[0,0], disp4[0,0])
    visualize6(imgL[0,0], imgR[0,0], disp1[0,1], disp2[0,1], disp3[0,1], disp4[0,1])
    
    # depth to disparity
    # dispL, dispR = depth_to_disp(depthL, depthR, camera_para)


    # left_ssim_1, right_ssim_1, left_l1_1, right_l1_1, lr_1, rl_1, smooth_left_disp_1, smooth_right_disp_1 = criteria(imgL1, imgR1, dispL1, dispR1)
    # left_ssim_2, right_ssim_2, left_l1_2, right_l1_2, lr_2, rl_2, smooth_left_disp_2, smooth_right_disp_2 = criteria(imgL2, imgR2, dispL2, dispR2)
    # left_ssim_3, right_ssim_3, left_l1_3, right_l1_3, lr_3, rl_3, smooth_left_disp_3, smooth_right_disp_3 = criteria(imgL3, imgR3, dispL3, dispR3)
    # left_ssim_4, right_ssim_4, left_l1_4, right_l1_4, lr_4, rl_4, smooth_left_disp_4, smooth_right_disp_4 = criteria(imgL4, imgR4, dispL4, dispR4)

    # print('left recons loss:', left_ssim_1.data.item(), 'right recons loss:', right_ssim_1.data.item(), \
    #     'left recons l1:', left_l1_1.data.item(), 'right recons l1:', right_l1_1.data.item(), \
    #     'lr consis loss:', lr_1.data.item(), 'rl consis loss:', rl_1.data.item(), \
    #      'smooth left disp:', smooth_left_disp_1, 'smooth right disp:', smooth_right_disp_1)


    # loss1 = 0.85*(left_ssim + right_ssim) + 0.15*(left_l1 + right_l1) + 1.0 * (lr_l1 + rl_l1)/(960.0*0.2) + 0.00 * (smooth_left_disp + smooth_right_disp)/(960.0*0.2)
    # torch.autograd.set_detect_anomaly(True)

    loss.backward()
    optimizer.step()
    
    return loss.data.item()

def val(imgL, imgR, model, optimizer, epoch):
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
        # dispL, dispR = depth_to_disp(depthL, depthR, camera_para)
        dispL, dispR = depthL.squeeze(1), depthR.squeeze(1)
        # reg = F.l1_loss(depthL) + F.l1_loss(depthR)
        left_ssim, right_ssim, left_l1, right_l1, lr_l1, rl_l1, smooth_left_disp, smooth_right_disp = criteria(imgL, imgR, dispL, dispR)
        loss = 0.85*(left_ssim + right_ssim) + 0.15*(left_l1 + right_l1) + 0.5 * (lr_l1 + rl_l1)/(960.0*0.2) + 0.05 * (smooth_left_disp + smooth_right_disp)/(960.0*0.2)

    return loss.data.item() 


def main():
    start_epoch = pasued_epoch+1
    total_epochs = args.epochs
    print('Start training ...')

    total_iteration = 0
    for epoch in range(start_epoch, total_epochs):
        print('\nEPOCH ' + str(epoch + 1) + ' of ' + str(total_epochs) + '\n')
        # -----------------Training---------------
        train_batch_num = 0
        epoch_train_loss = 0
        
        for idx, (left, right) in enumerate(tqdm(trainImgLoader)):
            start_time = time.time()
            train_loss = train(left, right, model, optimizer)
            epoch_train_loss += train_loss
            train_batch_num += 1
            total_iteration += 1
            print('Iter %d training loss = %.3f, time =  %.2f' %(idx, train_loss, time.time() - start_time))
            writer.add_scalar('Iteration loss', train_loss, total_iteration+1)

        epoch_train_loss /= train_batch_num
        writer.add_scalar('Loss/train', epoch_train_loss, epoch+1)


        #----------------Validation--------------
        # val_batch_num = 0
        # epoch_val_loss = 0
        # for idx, (left, right) in enumerate(tqdm(valImgLoader)):
        #     start_time = time.time()
        #     val_loss = val(left, right, model, optimizer, epoch)
        #     epoch_val_loss += val_loss
        #     val_batch_num += 1
        #     print('Iter %d val loss = %.3f, time =  %.2f' %(idx, val_loss, time.time() - start_time))

        # epoch_val_loss /= val_batch_num
        # writer.add_scalar('Loss/val', epoch_val_loss, epoch+1)

        savefilename = args.savemodel + 'Res18_md_' + str(epoch+1) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss':epoch_train_loss,
            # 'val_loss':epoch_val_loss,
            'optimizer':optimizer,
            'scheduler':scheduler
        }, savefilename)

        scheduler.step()
        lr = scheduler.get_lr()

    writer.close()

if __name__ == '__main__':
    main()
    # test()