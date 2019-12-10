# Author: Pengfei Li
# customed loss function, extension of torch.Module
# Author: Pengfei Li
# customed loss function, extension of torch.Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.disp_utils import reconstruct_left, reconstruct_right, SSIM, consistent_lr
class Loss_reonstruct(nn.Module):
    def __init__(self, n=4, h=1024, w=1280, default_device="cuda:0"):
        
        super(Loss_reonstruct, self).__init__()
        self.param = 0
        self.n, self.h, self.w = n, h, w
        self.data_type, self.device = torch.float32, default_device
        ## n, h, w = left_disp.shape
        grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=self.data_type), torch.arange(-1,1,2/w, dtype=self.data_type))
        self.base_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(self.device).requires_grad_(False)


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
        
        if n != self.n or w != self.w or h != self.h:
            grid_u, grid_v = torch.meshgrid(torch.arange(-1, 1, 2/h, dtype=data_type), torch.arange(-1,1,2/w, dtype=data_type))
            self.base_grid = torch.cat([grid_v.repeat([n,1,1]).unsqueeze(3), grid_u.repeat([n,1,1]).unsqueeze(3)], 3).to(device).requires_grad_(False)
            self.n, self.h, self.w = n, h, w
        if self.data_type != data_type or self.device != device:
            self.base_grid = self.base_grid.type(data_type).to(device)
            self.data_type, self.device = data_type, device

        left_recons = reconstruct_left(right_data, left_disp, left_grid=self.base_grid)
        right_recons = reconstruct_right(left_data, right_disp)

        res_loss_l = SSIM(left_recons, left_data)
        res_loss_r = SSIM(right_recons, right_data)
        loss_lr = consistent_lr(left_disp, right_disp, left_grid=self.base_grid)

        left_l1 = torch.abs(res_loss_l).mean()
        right_l1 = torch.abs(res_loss_r).mean()
        lr_l1 = torch.abs(loss_lr).mean()

        return left_l1, right_l1 , lr_l1

if __name__ == "__main__":
    from utils.disp_utils import load_exmaple
    import time 
    from torch.autograd import Variable

    disp_dir = "data/disp"
    img_dir = "data/sample"
    num_ins = 10
    right_data, left_data, left_disp, right_disp = load_exmaple(img_dir, disp_dir, num_ins, use_gpu=True)

    loss = Loss_reonstruct()
    for i in range(5):
        start = time.process_time()
        left_l1, right_l1 , lr_l1 = loss(left_data, right_data, left_disp, right_disp)
        overall_loss = left_l1 + right_l1 + lr_l1
        print(time.process_time() - start)
        overall_loss.backward()
        print("OK")
