import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, utils

from utils.datasets import *
from utils.functions import *
from trainer import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_models/143_enc.pth', help='pretrained stylegan2 model')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan2 model')
parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained ArcFace model')
parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--checkpoint_noiser', type=str, default='', help='checkpoint file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--input_path', type=str, default='./test/', help='evaluation data file path')
parser.add_argument('--save_path', type=str, default='./output/image/', help='output data save path')

opts = parser.parse_args()

log_dir = os.path.join(opts.log_path, opts.config) + '/'
config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)

# Initialize trainer
trainer = Trainer(config, opts)
trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)  
trainer.to(device)

state_dict = torch.load(opts.pretrained_model_path)#os.path.join(opts.log_path, opts.config + '/checkpoint.pth'))
trainer.enc.load_state_dict(torch.load(opts.pretrained_model_path))
trainer.enc.eval()

img_to_tensor = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# simple inference
image_dir = opts.input_path
save_dir = opts.save_path
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    img_list = [glob.glob1(image_dir, ext) for ext in ['*jpg','*png']]
    img_list = [item for sublist in img_list for item in sublist]
    img_list.sort()
    for i, img_name in enumerate(img_list):
        #print(i, img_name)
        image_A = img_to_tensor(Image.open(image_dir + img_name)).unsqueeze(0).to(device)
        output = trainer.test(img=image_A, return_latent=True)
        feature = output.pop()
        latent = output.pop()
        #np.save(save_dir + 'latent_code_%d.npy'%i, latent.cpu().numpy())
        utils.save_image(clip_img(output[1]), save_dir + img_name)
        if i > 1000:
            break
