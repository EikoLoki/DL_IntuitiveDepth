# Author: Pengfei Li
# customed loss function, extension of torch.Module
# Author: Pengfei Li
# customed loss function, extension of torch.Module
import torch
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

        return F.l1_loss(res_loss_r) + F.l1_loss(res_loss_l) + F.l1_loss(loss_lr)

