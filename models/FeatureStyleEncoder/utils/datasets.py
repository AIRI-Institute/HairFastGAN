import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torchvision import transforms, utils

class MyDataSet(data.Dataset):
    def __init__(self, image_dir=None, label_dir=None, output_size=(256, 256), noise_in=None, training_set=True, video_data=False, train_split=0.9):
        self.image_dir = image_dir
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Compose([
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])
        self.noise_in = noise_in
        self.video_data = video_data
        self.random_rotation = transforms.Compose([
            transforms.Resize(output_size),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.ToTensor()
        ])

        # load image file
        train_len = None
        self.length = 0
        self.image_dir = image_dir
        if image_dir is not None:
            img_list = [glob.glob1(self.image_dir, ext) for ext in ['*jpg','*png']]
            image_list = [item for sublist in img_list for item in sublist]
            image_list.sort()
            train_len = int(train_split*len(image_list))
            if training_set:
                self.image_list = image_list[:train_len]
            else:
                self.image_list = image_list[train_len:]
            self.length = len(self.image_list)

        # load label file
        self.label_dir = label_dir
        if label_dir is not None:
            self.seeds = np.load(label_dir)
            if train_len is None:
                train_len = int(train_split*len(self.seeds))
            if training_set:
                self.seeds = self.seeds[:train_len]
            else:
                self.seeds = self.seeds[train_len:]
            if self.length == 0:
                self.length = len(self.seeds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = None
        if self.image_dir is not None:
            img_name = os.path.join(self.image_dir, self.image_list[idx])
            image = Image.open(img_name)
            img = self.resize(image)
            if img.size(0) == 1:
                img = torch.cat((img, img, img), dim=0)
            img = self.normalize(img)

        # generate image 
        if self.label_dir is not None:
            torch.manual_seed(self.seeds[idx])
            z = torch.randn(1, 512)[0]
            if self.noise_in is None:
                n = [torch.randn(1, 1)]
            else:
                n = [torch.randn(noise.size())[0] for noise in self.noise_in]
            if img is None:
                return z, n 
            else:
                return z, img, n
        else:
            return img

class Car_DataSet(data.Dataset):
    def __init__(self, image_dir=None, label_dir=None, output_size=(512, 512), noise_in=None, training_set=True, video_data=False, train_split=0.9):
        self.image_dir = image_dir
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.resize = transforms.Compose([
            transforms.Resize((384, 512)),
            transforms.Pad(padding=(0, 64, 0, 64)),
            transforms.ToTensor()
        ])
        self.noise_in = noise_in
        self.video_data = video_data
        self.random_rotation = transforms.Compose([
            transforms.Resize(output_size),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.ToTensor()
        ])

        # load image file
        train_len = None
        self.length = 0
        self.image_dir = image_dir
        if image_dir is not None:
            img_list = [glob.glob1(self.image_dir, ext) for ext in ['*jpg','*png']]
            image_list = [item for sublist in img_list for item in sublist]
            image_list.sort()
            train_len = int(train_split*len(image_list))
            if training_set:
                self.image_list = image_list[:train_len]
            else:
                self.image_list = image_list[train_len:]
            self.length = len(self.image_list)

        # load label file
        self.label_dir = label_dir
        if label_dir is not None:
            self.seeds = np.load(label_dir)
            if train_len is None:
                train_len = int(train_split*len(self.seeds))
            if training_set:
                self.seeds = self.seeds[:train_len]
            else:
                self.seeds = self.seeds[train_len:]
            if self.length == 0:
                self.length = len(self.seeds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = None
        if self.image_dir is not None:
            img_name = os.path.join(self.image_dir, self.image_list[idx])
            image = Image.open(img_name)
            img = self.resize(image)
            if img.size(0) == 1:
                img = torch.cat((img, img, img), dim=0)
            img = self.normalize(img)
            if self.video_data:
                img_2 = self.random_rotation(image)
                img_2 = self.normalize(img_2)
                img_2 = torch.where(img_2 > -1, img_2, img)
                img = torch.cat([img, img_2], dim=0)

        # generate image 
        if self.label_dir is not None:
            torch.manual_seed(self.seeds[idx])
            z = torch.randn(1, 512)[0]
            n = [torch.randn_like(noise[0]) for noise in self.noise_in]
            if img is None:
                return z, n 
            else:
                return z, img, n
        else:
            return img

