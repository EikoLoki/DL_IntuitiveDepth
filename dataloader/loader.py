from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import utils.camera_config as cam_config

def image_loader(file):
    return Image.open(file).convert('RGB')

def camera_config_loader(file):
    cam_para = cam_config.CameraPara(file)
    return cam_para


class SCARED_loader(data.Dataset):
    def __init__(self, left, right, camera_para, training=True, imgloader=image_loader, paraloader=camera_config_loader):
        self.left = left
        self.right = right
        self.camera_para = camera_para
        self.img_loader = imgloader
        self.para_loader = paraloader
        self.training = training

    def __getitem__(self, index):
        left_file = self.left[index]
        right_file = self.right[index]
        para_file = self.camera_para[index]

        left_img = self.img_loader(left_file)
        right_img = self.img_loader(right_file)
        para = self.para_loader(para_file)

        # TODO: use preprosssing function here
        if self.training:
            preprocess = transforms.Compose([
                transforms.ToTensor()])
        else:
            # no preprocess, just toTensor here
            preprocess = transforms.Compose([
                transforms.ToTensor()])


        left = preprocess(left_img)
        right = preprocess(right_img)
        para_dict = {
            'left_intrinsic': para.l_camera_intrin,
            'right_intrinsic': para.r_camera_intrin,
            'rotation':para.rotation,
            'translation':para.translation}



        return left, right, para_dict

    def __len__(self):
        return len(self.left)

