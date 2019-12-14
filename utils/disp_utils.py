import re
import numpy as np
import sys
import cv2 
import os 
from os.path import join 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

import time

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    # header = header[2:4]
    # header = bytes(header)
    # print(header)
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    #data = np.flipud(data)
    return data, scale

    img_right = cv2.imread(join(img_dir, "right_finalpass/frame_data{:>06}.png".format(indx)) )
    img_right = cv2.imread(join(img_dir, "right_finalpass/frame_data{:>06}.png".format(indx)) )

def visualize(img_left, img_right, img_recons, data):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1,4)
    axes[0].imshow(img_left)
    axes[1].imshow(img_right)
    axes[2].imshow(img_recons)
    axes[3].imshow((data+1e-4), cmap="plasma")
    plt.show()
    # input("Any key to continue")
    # print("OK")

def visualize6(img_l, img_r, disp1, disp2, disp3, disp4):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2,3)
    axes[0,0].imshow(img_l)
    axes[0,1].imshow(img_r)
    axes[0,2].imshow(disp1, cmap="plasma")
    axes[1,0].imshow(disp2, cmap="plasma")
    axes[1,1].imshow(disp3, cmap="plasma")
    axes[1,2].imshow(disp4, cmap="plasma")
    plt.show()
    # input("Any key to continue")
    # print("OK")

def SSIM(x, y, ksize = 9):
    
    # C1 = 0.01 ** 2
    # C2 = 0.03 ** 2
    # ps = ksize//2
    # mu_x = F.avg_pool2d(x, kernel_size=ksize, stride=1, padding=ps)
    # mu_y = F.avg_pool2d(y, kernel_size=ksize, stride=1, padding=ps)
    
    # sigma_x  = F.avg_pool2d(x * x, kernel_size=ksize, stride=1, padding=ps) - mu_x.pow(2)
    # sigma_y  = F.avg_pool2d(y * y, kernel_size=ksize, stride=1, padding=ps) - mu_y.pow(2)
    # sigma_xy = F.avg_pool2d(x * x, kernel_size=ksize, stride=1, padding=ps) - mu_x * mu_y

    # SSIM_d = ((mu_x * mu_x + mu_y * mu_y) + C1) * (sigma_x + sigma_y + C2)
    # #del sigma_x, sigma_y
    # SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    # #del sigma_xy
    # loss_SSIM = SSIM_n / SSIM_d

    # return torch.clamp((1 - loss_SSIM) / 2, 0, 1)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
    mu_y = nn.AvgPool2d(3, 1, padding=1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, padding=1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, padding=1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)
    

def reconstruct_left(right_data, left_disp, left_grid=None):
    '''
    Args:
        right data [N, C, H, W]
        left disp [N, H, W]
    Return:
        right_recons [N, C, H, W]
    '''
    n, _, h, w = right_data.shape
    device = right_data.device
    data_type = right_data.dtype
    if left_grid is None:
        grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=data_type), torch.arange(-1,1,2/w, dtype=data_type))
        left_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(device).requires_grad_(False)
    
    grid_sample = left_grid + torch.cat([-2*left_disp.unsqueeze(-1)/w , torch.zeros(n,h,w,1, device=device, dtype=data_type)], 3)

    left_recons = F.grid_sample(right_data, grid_sample, mode='bilinear', padding_mode='zeros')

    return left_recons

def reconstruct_right(left_data, right_disp, right_grid=None):
    '''
    Args:
        left data [N, C, H, W]
        right disp [N, H, W]
    Return:
        right_recons [N, C, H, W]
    '''
    #assert 0
    n, _, h, w = left_data.shape
    device = left_data.device
    data_type = left_data.dtype

    if right_grid is None:
        grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=data_type), torch.arange(-1,1,2/w, dtype=data_type))
        right_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(device).requires_grad_(False)
    
    grid_sample = right_grid + torch.cat([ 2*right_disp.unsqueeze(-1)/w , torch.zeros(n,h,w,1, device=device, dtype=data_type)], 3)
    
    right_recons = F.grid_sample(left_data, grid_sample, mode='bilinear', padding_mode='zeros')

    return right_recons

def apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output

def reconstruct_left_monodepth(right_img, left_disp, left_grid=None):
    left_disp = left_disp.unsqueeze(1)
    recons_left = apply_disparity(right_img, -left_disp)
    return recons_left

def reconstruct_right_monodepth(left_img, right_disp, right_grid=None):
    right_disp = right_disp.unsqueeze(1)
    recons_right = apply_disparity(left_img, right_disp)
    return recons_right

def consistent_lr_monodepth(left_disp, right_disp, left_grid=None):
    right_disp = right_disp.unsqueeze(1)
    left_disp = left_disp.unsqueeze(1)
    recons_disp_left = apply_disparity(right_disp, -left_disp)
    lr_loss = torch.mean(torch.abs(left_disp - recons_disp_left))
    return lr_loss

def consistent_rl_monodepth(left_disp, right_disp, right_grid=None):
    right_disp = right_disp.unsqueeze(1)
    left_disp = left_disp.unsqueeze(1)
    recons_disp_right = apply_disparity(left_disp, right_disp)
    rl_loss = torch.mean(torch.abs(left_disp - recons_disp_right))
    return rl_loss



def consistent_lr(left_disp, right_disp, left_grid = None):
    '''
    Args:
        left_disp [N, H, W]
        right_disp [N, H, W]
    Return:
        lr_loss [N, H, W]
    '''

    n, h, w = left_disp.shape
    device = left_disp.device
    data_type = left_disp.dtype

    if left_grid is None:
        grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=data_type), torch.arange(-1,1,2/w, dtype=data_type))
        left_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(device).requires_grad_(False)
    
    # left_grid[:,:,:,1] = grid_u.repeat([n,1,1])
    # left_grid[:,:,:,0] = grid_v.repeat([n,1,1])
    # left_grid[:,:,:,0] -= 2*left_disp/w 
    grid_sample = left_grid + torch.cat([-2*left_disp.unsqueeze(-1)/w, torch.zeros(n,h,w,1, device=device, dtype=data_type)], 3)
    
    left_disp_recons = F.grid_sample(right_disp.unsqueeze(1), grid_sample, mode='bilinear', padding_mode='zeros')
    
    return left_disp_recons.squeeze() - left_disp, left_disp_recons

def consistent_rl(left_disp, right_disp, right_grid = None):
    '''
    Args:
        left_disp [N, H, W]
        right_disp [N, H, W]
    Return:
        lr_loss [N, H, W]
    '''

    n, h, w = left_disp.shape
    device = left_disp.device
    data_type = left_disp.dtype

    if right_grid is None:
        grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=data_type), torch.arange(-1,1,2/w, dtype=data_type))
        right_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(device).requires_grad_(False)
    
    # left_grid[:,:,:,1] = grid_u.repeat([n,1,1])
    # left_grid[:,:,:,0] = grid_v.repeat([n,1,1])
    # left_grid[:,:,:,0] -= 2*left_disp/w 
    grid_sample = right_grid + torch.cat([2*right_disp.unsqueeze(-1)/w, torch.zeros(n,h,w,1, device=device, dtype=data_type)], 3)
    
    right_disp_recons = F.grid_sample(left_disp.unsqueeze(1), grid_sample, mode='bilinear', padding_mode='zeros')

    
    return right_disp_recons.squeeze() - right_disp, right_disp_recons


def depth_to_disp(depthL, depthR, camera_para):
    """
    Args:
        depthL, depthR(torch.Tensor): depth map for Left view and Right view
        camera_para(dict): camera parameters for left and right.
    Returns:
        dispL, dispR: disparity of left view and right view
    """
    l_intrinsic = torch.tensor(camera_para['left_intrinsic'])
    r_intrinsic = torch.tensor(camera_para['right_intrinsic'])
    translation = torch.tensor(camera_para['translation'])

    l_f = ((l_intrinsic[:,0,0]+l_intrinsic[:,1,1])/2).view(-1,1,1,1).cuda()
    r_f = ((r_intrinsic[:,0,0]+r_intrinsic[:,1,1])/2).view(-1,1,1,1).cuda()
    bl = torch.abs(translation[:,0,0]).view(-1,1,1,1).cuda()
    # print('depth size:', depthL.size())
    # print('left focal length size:', l_f.size())


    # maskL = (depthL > 0.001)
    # maskR = (depthR > 0.001)
    # maskL.detach_()
    # maskR.detach_()
    # depthL = depthL*maskL.int().float()
    # depthR = depthR*maskR.int().float()

    dispL = l_f * bl / depthL
    dispR = r_f * bl / depthR

    # print(dispL)

    dispL = dispL.squeeze(1).type(torch.float32)
    dispR = dispR.squeeze(1).type(torch.float32)
    dispL.requires_grad_ = True
    dispR.requires_grad_ = True
    return dispL, dispR



def load_exmaple(img_dir, disp_dir, num_ins):

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    disp_l, scale = readPFM( join(disp_dir, "left/frame_data{:>06}.pfm".format(0)) )
    h, w = disp_l.shape
    right_data = np.empty([num_ins, 3, h, w], dtype=np.float32)
    left_data  = np.empty([num_ins, 3, h, w], dtype=np.float32)
    left_disp  = np.empty([num_ins, h, w], dtype=np.float32)
    right_disp = np.empty([num_ins, h, w], dtype=np.float32)

    for i in range(num_ins):
        indx = i
        disp_l, _ = readPFM( join(disp_dir, "left/frame_data{:>06}.pfm".format(indx)) )
        disp_r, _ = readPFM( join(disp_dir, "right/frame_data{:>06}.pfm".format(indx)) )

        #disp_l = cv2.GaussianBlur(disp_l,(5,5),0)
        img_left  = cv2.imread(join(img_dir, "left_finalpass/frame_data{:>06}.png".format(indx)))[:,:,::-1]
        img_right = cv2.imread(join(img_dir, "right_finalpass/frame_data{:>06}.png".format(indx)) )[:,:,::-1]

        #visualize(img_left, img_right, disp_r)
        right_data[i, :, :, :] = np.transpose(img_right/255.0,(2,0,1))
        left_data[i, :, :, :] = np.transpose(img_left/255.0,(2,0,1))
        right_disp[i, :, :] = disp_r
        left_disp[i, :, :] = disp_l
        

    right_img = torch.from_numpy(right_data).to(device).requires_grad_()
    left_img = torch.from_numpy(left_data).to(device).requires_grad_()
    right_disp_gd = torch.from_numpy(right_disp).to(device).requires_grad_()
    left_disp_gd = torch.from_numpy(left_disp).to(device).requires_grad_()

    return right_img, left_img, right_disp_gd, left_disp_gd


if __name__ == "__main__":
    
    disp_dir = "data/disp"
    img_dir = "data/sample"
    num_ins = 4

    right_data, left_data, left_disp, right_disp = load_exmaple(img_dir, disp_dir, num_ins)

    start = time.process_time()
    left_recons = reconstruct_left(right_data, left_disp)
    right_recons = reconstruct_right(left_data, right_disp)
    print(time.process_time() - start)
    res_loss_r = SSIM(right_recons, right_data)
    res_loss_l = SSIM(left_recons, left_data)
    loss_lr = consistent_lr(left_disp, right_disp)
    
    print(time.process_time() - start)

    

    # indx = 0
    # img_r_res = right_recons.numpy()[indx, :,:,:]
    # img_r_res = np.transpose(img_r_res, (1,2,0))
    # img_right = right_data.numpy()[indx, :,:,:]
    # img_right = np.transpose(img_right, (1,2,0))
    # loss_vis_r = np.linalg.norm(res_loss_r[indx, :, :, :].numpy().squeeze(), axis=0)
    # loss_lr_i = loss_lr[indx, :, :].numpy()
    # visualize(img_right, img_r_res, loss_lr_i)
    
    # indx = 0
    # img_l_res = left_recons.numpy()[indx, :,:,:]
    # img_l_res = np.transpose(img_l_res, (1,2,0))
    # img_left = left_data.numpy()[indx, :,:,:]
    # img_left = np.transpose(img_left, (1,2,0))
    # loss_vis_l = np.mean(res_loss_l[indx, :, :, :].numpy().squeeze(), axis=0)
    # visualize(img_left, img_l_res, loss_vis_l)