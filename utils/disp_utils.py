import re
import numpy as np
import sys
import cv2 
import os 
from os.path import join 
import torch
import torch.nn.functional as F

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

def visualize(img_left, img_right, data):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(img_left)
    axes[1].imshow(img_right)
    axes[2].imshow((data+1e-4), cmap="plasma")
    fig.show()
    input("Any key to continue")
    print("OK")

def SSIM(x, y, ksize = 3):
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ps = ksize//2
    mu_x = F.avg_pool2d(x, kernel_size=ksize, stride=1, padding=ps)
    mu_y = F.avg_pool2d(y, kernel_size=ksize, stride=1, padding=ps)
    
    sigma_x  = F.avg_pool2d(x * x, kernel_size=ksize, stride=1, padding=ps) - mu_x.pow(2)
    sigma_y  = F.avg_pool2d(y * y, kernel_size=ksize, stride=1, padding=ps) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * x, kernel_size=ksize, stride=1, padding=ps) - mu_x * mu_y

    SSIM_d = ((mu_x * mu_x + mu_y * mu_y) + C1) * (sigma_x + sigma_y + C2)
    #del sigma_x, sigma_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    #del sigma_xy
    loss_SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - loss_SSIM) / 2, 0, 1)
    

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
    
    return left_disp_recons.squeeze() - left_disp

def load_exmaple(img_dir, disp_dir, num_ins, use_gpu=True):

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