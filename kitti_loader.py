'''
This file implements the data loader for LidarStereoNet's training phase.
Note that images are randomly cropped to 256x512 dimensions and normalized
according to the imagenet stats. Credit: PSMNet code. Note that without
cropping, the GPU's memory (K80 GPU with 11GB memory) wouldn't be enough.
'''

import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2 as cv
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def image_loader(path):
    #return Image.open(path).convert('RGB')
    #return Image.open(path)
    img = cv.imread(path, cv.IMREAD_UNCHANGED).astype(np.float)
    return img

def to_tensor(x):
    width, height = x.size
    tensor = torch.empty(3, width, height).uniform_(0, 1)
    trans = transforms.ToPILImage()(tensor)
    return tensor

def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)

class KittiLoader(data.Dataset):
    def __init__(self, left, right, lidar_left, lidar_right, training, loader=image_loader, transform=None):

        self.left = left
        self.right = right
        # self.disp_L = left_disparity
        self.lidar_left = lidar_left
        self.lidar_right = lidar_right
        self.loader = loader
        self.training = training
        self.transform = transform

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        lidar_left = self.lidar_left[index]
        lidar_right = self.lidar_right[index]
        # lidar_right = np.fromfile(self.lidar_right[index], dtype=np.float32).reshape(-1,4)
        # disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        lidar_left_img = self.loader(lidar_left)
        lidar_right_img = self.loader(lidar_right)
        # dataL, scaleL = self.dploader(disp_L)
        # dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        if self.training:
            h, w = left_img.shape[0:2]
            th, tw = 256, 512

            y1 = random.randint(0, w - tw)
            x1 = random.randint(0, h - th)

            left_img = left_img[x1:x1+th, y1:y1+tw, :]
            right_img = right_img[x1:x1+th, y1:y1+tw, :]
            lidar_left_img = lidar_left_img[x1:x1+th, y1:y1+tw]
            lidar_right_img = lidar_right_img[x1:x1+th, y1:y1+tw]

        else:
            h, w = left_img.shape[0:2]

            left_img = left_img[h-368:h, w-1232:w, :]
            right_img = right_img[h-368:h, w-1232:w, :]
            lidar_left_img = lidar_left_img[h-368:h, w-1232:w]
            lidar_right_img = lidar_right_img[h-368:h, w-1232:w]

        left_img = torch.tensor(left_img).float().permute([2,0,1])
        right_img = torch.tensor(right_img).float().permute([2,0,1])
        left_img = scale_crop(256)(left_img)
        right_img = scale_crop(256)(right_img)
        lidar_left_img = torch.tensor(lidar_left_img).float().unsqueeze(dim=0) / 256
        lidar_right_img = torch.tensor(lidar_right_img).float().unsqueeze(dim=0) / 256

        return left_img, right_img, lidar_left_img, lidar_right_img

    def __len__(self):
        return len(self.left)