import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from torchvision import transforms, utils
from tensorboard_logger import Logger
from tqdm import tqdm
from utils.functions import *

import sys
sys.path.append('pixel2style2pixel/')
from pixel2style2pixel.models.stylegan2.model import Generator, get_keys

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='002', help='Path to the config file.')
parser.add_argument('--dataset_path', type=str, default='./data/stylegan2-generate-images/', help='dataset path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan model')
opts = parser.parse_args()


StyleGAN = Generator(1024, 512, 8)
state_dict = torch.load(opts.stylegan_model_path, map_location='cpu')
StyleGAN.load_state_dict(get_keys(state_dict, 'decoder'), strict=True)
StyleGAN.to(device)

#seeds = np.array([torch.random.seed() for i in range(100000)])
seeds = np.load(opts.dataset_path + 'seeds_pytorch_1.8.1.npy')

with torch.no_grad():
    os.makedirs(opts.dataset_path + 'ims/', exist_ok=True)

    for i, seed in enumerate(tqdm(seeds)):

        torch.manual_seed(seed)
        z = torch.randn(1, 512).to(device)
        n = StyleGAN.make_noise()
        w = StyleGAN.get_latent(z)
        x, _ = StyleGAN([w], input_is_latent=True, noise=n)
        utils.save_image(clip_img(x), opts.dataset_path + 'ims/%06d.jpg'%i)
