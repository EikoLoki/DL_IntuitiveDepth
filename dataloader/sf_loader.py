import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import dataloader.sf_listfile as lt
import numpy as np
import dataloader.sf_preprocess as preprocess
import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(data.Dataset):
    def __init__(self, left, right, training, loader=default_loader):
        self.left = left
        self.right = right
        self.loader = loader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]


        left_img = self.loader(left)
        right_img = self.loader(right)

        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img
        else:
           w, h = 896, 512
           left_img = left_img.crop((0, 0, w, h))
           right_img = right_img.crop((0, 0, w, h))
           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img

    def __len__(self):
        return len(self.left)
