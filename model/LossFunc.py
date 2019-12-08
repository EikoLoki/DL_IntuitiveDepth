# Author: Pengfei Li
# customed loss function, extension of torch.Module
# Author: Pengfei Li
# customed loss function, extension of torch.Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.disp_utils import reconstruct_left, reconstruct_right, SSIM, consistent_lr
class Loss_reonstruct(nn.Module):
    def __init__(self):
        super(Loss_reonstruct, self).__init__()
        self.param = 0

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
        left_recons = reconstruct_left(right_data, left_disp)
        right_recons = reconstruct_right(left_data, right_disp)
        res_loss_r = SSIM(right_recons, right_data)
        res_loss_l = SSIM(left_recons, left_data)
        loss_lr = consistent_lr(left_disp, right_disp)

        right_l1 = torch.abs(res_loss_r).mean()
        left_l1 = torch.abs(res_loss_l).mean()
        lr_l1 = torch.abs(loss_lr).mean()

        return right_l1 + left_l1 + lr_l1

if __name__ == "__main__":
    from utils.disp_utils import load_exmaple
    disp_dir = "data/disp"
    img_dir = "data/sample"
    num_ins = 8
    right_data, left_data, left_disp, right_disp = load_exmaple(img_dir, disp_dir, num_ins)

    loss = Loss_reonstruct()
    
    output = loss(left_data, right_data, left_disp, right_disp)
    #print(output)
    print("OK")