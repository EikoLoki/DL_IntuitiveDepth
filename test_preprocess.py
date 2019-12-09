import dataloader.listfile as lf
import dataloader.loader as ld
import torch
import torch.utils.data

if __name__ == '__main__':
    train_left_img, train_right_img, train_cam_para, \
    val_left_img, val_right_img, val_cam_para, \
    test_left_img, test_right_img, test_cam_para = lf.SCARED_lister('data')

    trainImgLoader = torch.utils.data.DataLoader(
        ld.SCARED_loader(train_left_img, train_right_img, train_cam_para, training=True),
        batch_size=5, shuffle=True, num_workers=5, drop_last=False
    )

    for idx, (left, right, para) in enumerate(trainImgLoader):
        print('batch index:', idx)
        print(left.size())
        print(len(para))