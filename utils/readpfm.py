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
    axes[2].imshow((data+1e-4), cmap="Greys")
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

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = ((mu_x * mu_x + mu_y * mu_y) + C1) * (sigma_x + sigma_y + C2)

    loss_SSIM = SSIM_n / SSIM_d
    print(loss_SSIM.shape)

    #return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    return (1 - loss_SSIM) / 2

def reconstruct_left(right_data, left_disp):

    n, _, h, w = right_data.shape
    device = right_data.device()

    grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=torch.double), torch.arange(-1,1,2/w, dtype=torch.double))
    left_grid = torch.empty(n, h, w, 2, dtype= torch.double, device=device)
    
    left_grid[:,:,:,1] = grid_u.repeat([n,1,1])
    left_grid[:,:,:,0] = grid_v.repeat([n,1,1])
    left_grid[:,:,:,0] -= 2*left_disp/w 
    
    left_recons = F.grid_sample(right_data, left_grid, mode='bilinear', padding_mode='zeros')

    return left_recons


if __name__ == "__main__":
    indx = 0
    disp_dir = "data/sample_disp"
    img_dir = "data/sample"
    num_ins = 1

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    disp_l, scale = readPFM( join(disp_dir, "frame_data{:>06}.pfm".format(0)) )
    h, w = disp_l.shape
    right_data = torch.empty(num_ins, 3, h, w, dtype=torch.double).to(device)
    left_data = torch.empty(num_ins, 3, h, w, dtype=torch.double).to(device)
    left_disp = torch.empty(num_ins, h, w, dtype=torch.double).to(device)

    for i in range(num_ins):
        disp_l, scale = readPFM( join(disp_dir, "frame_data{:>06}.pfm".format(indx)) )
        #disp_l = cv2.GaussianBlur(disp_l,(5,5),0)
        img_left  = cv2.imread(join(img_dir, "left_finalpass/frame_data{:>06}.png".format(indx)))[:,:,::-1]
        img_right = cv2.imread(join(img_dir, "right_finalpass/frame_data{:>06}.png".format(indx)) )[:,:,::-1]
        right_data[i, :, :, :] = torch.from_numpy(np.transpose(img_right/255.0,(2,0,1)))
        left_data[i, :, :, :] = torch.from_numpy(np.transpose(img_left/255.0,(2,0,1)))
        left_disp[i, :, :] = torch.from_numpy(disp_l)

    
    #visualize(img_left, img_right, disp_l)
    
    # right_data = torch.from_numpy(np.transpose(img_right/255.0,(2,0,1))).unsqueeze(0)
    # left_data = torch.from_numpy(np.transpose(img_left/255.0,(2,0,1))).unsqueeze(0)
    # left_disp = torch.from_numpy(disp_l.astype(np.double))
    start = time.process_time()

    left_recons = reconstruct_left(right_data, left_disp)
    print(time.process_time() - start)
    res_loss = SSIM(left_recons, left_data)
    print(time.process_time() - start)
    # print(res_loss.shape)
    # img_l_res = left_recons.numpy().squeeze()
    # img_l_res = np.transpose(img_l_res, (1,2,0))
    # loss_vis = np.mean(res_loss.numpy().squeeze(), axis=0)

    # indx = 0
    # img_l_res = left_recons.numpy()[indx, :,:,:]
    # img_l_res = np.transpose(img_l_res, (1,2,0))
    # img_left = left_data.numpy()[indx, :,:,:]
    # img_left = np.transpose(img_left, (1,2,0))
    # loss_vis = np.mean(res_loss[indx, :, :, :].numpy().squeeze(), axis=0)
    
    # visualize(img_left, img_l_res, loss_vis)
    