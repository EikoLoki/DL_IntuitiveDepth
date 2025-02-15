# Author: Pengfei Li
# customed loss function, extension of torch.Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.disp_utils import reconstruct_left, reconstruct_right, SSIM, consistent_lr, consistent_rl, \
reconstruct_left_monodepth, reconstruct_right_monodepth,consistent_lr_monodepth, consistent_rl_monodepth,visualize

class Loss_reonstruct(nn.Module):
    def __init__(self, n=4, h=1024, w=1280, default_device="cuda:0"):
        
        super(Loss_reonstruct, self).__init__()
        self.param = 0
        self.n, self.h, self.w = n, h, w
        self.data_type, self.device = torch.float32, default_device
        ## n, h, w = left_disp.shape
        grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=self.data_type), torch.arange(-1,1,2/w, dtype=self.data_type))
        self.base_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(self.device).requires_grad_(False)

    def gradient_x(self, img):

        # Pad input to keep output size consistent
        if len(img.shape) == 4:
            img = F.pad(img (0, 1, 0, 0), mode="replicate")
        elif len(img.shape) == 3:
            img = F.pad(img.unsqueeze(1), (0, 1, 0, 0), mode="replicate")
        
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):

        # Pad input to keep output size consistent
        if len(img.shape) == 4:
            img = F.pad(img (0, 0, 0, 1), mode="replicate")
        elif len(img.shape) == 3:
            img = F.pad(img.unsqueeze(1), (0, 0, 0, 1), mode="replicate")
        
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def smoothness_term(self, img_data):

        gx = self.gradient_x(img_data)
        gy = self.gradient_y(img_data)

        return torch.abs(gx) + torch.abs(gy)

    def forward(self, left_data, right_data, left_disp, right_disp):
        '''
        Loss function for the disparity, consists of reconstruction loss for left and right, and left-right disparity consistency
        Args:
            left_data  [N, C, H, W]
            right_data [N, C, H, W]
            left_disp  [N, H, W]
            right_disp [N, H, W]
        Return:
            lr_loss [N, H, W]
        '''

        n, h, w = right_disp.shape
        data_type = right_disp.dtype
        device = right_data.device
        # print('dtype:', data_type)
        # print('device:', device)
        
        if n != self.n or w != self.w or h != self.h:
            grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=data_type), torch.arange(-1,1,2/w, dtype=data_type))
            self.base_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(device).requires_grad_(False)
            self.n, self.h, self.w = n, h, w
        if self.data_type != data_type or self.device != device:
            self.base_grid = self.base_grid.type(data_type).to(device)
            self.data_type, self.device = data_type, device
        
        # old version writen by Pengfei
        left_recons = reconstruct_left(right_data, left_disp, left_grid=self.base_grid)
        right_recons = reconstruct_right(left_data, right_disp)

        # new version from monodepth
        # left_recons = reconstruct_left_monodepth(right_data, left_disp, left_grid=self.base_grid)
        # right_recons = reconstruct_right_monodepth(left_data, right_disp, right_grid=self.base_grid)
        
        res_loss_l = SSIM(left_recons, left_data)
        left_ssim = torch.abs(res_loss_l).mean()
        left_l1 = F.l1_loss(res_loss_l, left_data)

        res_loss_r = SSIM(right_recons, right_data)
        right_ssim = torch.abs(res_loss_r).mean()
        right_l1 = F.l1_loss(right_recons, right_data)

        # old version writen by Pengfei
        loss_lr, left_disp_recons = consistent_lr(left_disp, right_disp, left_grid=self.base_grid)
        loss_rl, right_disp_recons = consistent_rl(left_disp, right_disp, right_grid=self.base_grid)
        lr_l1 = torch.abs(loss_lr).mean()
        rl_l1 = torch.abs(loss_rl).mean()

        # visualization
        left_recons_cpu = left_recons.cpu()
        right_recons_cpu = right_recons.cpu()
        left_data_cpu = left_data.cpu()
        right_data_cpu = right_data.cpu()
        left_disp_cpu = left_disp.cpu()
        right_disp_cpu = right_disp.cpu()
        left_disp_recons_cpu = left_disp_recons.cpu()
        right_disp_recons_cpu = right_disp_recons.cpu()
        loss_lr_cpu = loss_lr.cpu()
        loss_rl_cpu = loss_rl.cpu()
        

        left_recons_cpu.detach_()
        right_recons_cpu.detach_()
        left_data_cpu.detach_()
        right_data_cpu.detach_()
        left_disp_cpu.detach_()
        right_disp_cpu.detach_()
        left_disp_recons_cpu.detach_()
        right_disp_recons_cpu.detach_()
        loss_lr_cpu.detach_()
        loss_rl_cpu.detach_()

        print(left_disp_cpu)
        visualize(left_data_cpu[0,0], right_data_cpu[0,0], left_recons_cpu[0,0], left_disp_cpu[0])
        visualize(left_data_cpu[0,0], right_data_cpu[0,0], right_recons_cpu[0,0], right_disp_cpu[0])
        visualize(left_disp_cpu[0], right_disp_cpu[0], left_disp_recons_cpu[0,0], loss_lr_cpu[0])
        visualize(left_disp_cpu[0], right_disp_cpu[0], right_disp_recons_cpu[0,0], loss_rl_cpu[0])
        
        # new version from Monodepth
        # lr_l1 = consistent_lr_monodepth(left_disp, right_disp, left_grid=self.base_grid)
        # rl_l1 = consistent_rl_monodepth(left_disp, right_disp, right_grid=self.base_grid)

        left_smooth  = self.smoothness_term(left_disp).mean()
        right_smooth = self.smoothness_term(right_disp).mean()

        return left_ssim, right_ssim, left_l1, right_l1 , lr_l1, rl_l1, left_smooth, right_smooth

if __name__ == "__main__":
    from utils.disp_utils import load_exmaple
    import time 
    from torch.autograd import Variable

    disp_dir = "data/disp"
    img_dir = "data/sample"
    num_ins = 4
    right_data, left_data, left_disp, right_disp = load_exmaple(img_dir, disp_dir, num_ins, use_gpu=True)

    loss = Loss_reonstruct()
    for i in range(5):
        #start = time.process_time()
        left_l1, right_l1 , lr_l1, left_smooth, right_smooth = loss(left_data, right_data, left_disp, right_disp)
        overall_loss = 0.5*left_l1 + 0.5*right_l1  + 1.0*lr_l1 + 0.5*left_smooth + 0.5*right_smooth
        print(overall_loss)
        overall_loss.backward()
        #print(overall_loss.grad)
        #print(left_data.grad)
