# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import copy
import glob
import numpy as np
import os
import torch
import yaml
import time 

from PIL import Image
from torchvision import transforms, utils, models

from utils.video_utils import *
from face_parsing.model import BiSeNet
from trainer import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--attr', type=str, default='Eyeglasses', help='attribute for manipulation.')
parser.add_argument('--alpha', type=str, default='1.', help='scale for manipulation.')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_models/143_enc.pth', help='pretrained stylegan2 model')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan2 model')
parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained ArcFace model')
parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--function', type=str, default='', help='Calling function by name.')
parser.add_argument('--video_path', type=str, default='./data/video/FP006911MD02.mp4', help='video file path')
parser.add_argument('--output_path', type=str, default='./output/video/', help='output video file path')
parser.add_argument('--boundary_path', type=str, default='./boundaries_ours/', help='output video file path')
parser.add_argument('--optical_flow', action='store_true', help='use optical flow')
parser.add_argument('--resize', action='store_true', help='downscale image size')
parser.add_argument('--seamless', action='store_true', help='seamless cloning')
parser.add_argument('--filter_size', type=float, default=3, help='filter size')
parser.add_argument('--strs', type=str, default='Original,Projected,Manipulated', help='strs to be added on video')
opts = parser.parse_args()

# Celeba attribute list
attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
            'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
            'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
            'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
            'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

img_to_tensor = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# linear interpolation
def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
                    len(boundary.shape) == 2 and
                    boundary.shape[1] == latent_code.shape[-1])

    linspace = np.linspace(start_distance, end_distance, steps)
    if len(latent_code.shape) == 2:
        linspace = linspace.reshape(-1, 1).astype(np.float32)
        return latent_code + linspace * boundary
    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + linspace * boundary.reshape(1, 1, -1)
        
# Latent code manipulation
def latent_manipulation(opts, align_dir_path, process_dir_path):
    
    os.makedirs(process_dir_path, exist_ok=True)
    #attrs = opts.attr.split(',')
    #alphas = opts.alpha.split(',')
    step_scale = 15 * int(opts.alpha)
    n_steps = 5
    
    boundary = np.load(opts.boundary_path +'%s_boundary.npy'%opts.attr)
    
    # Initialize trainer
    config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)
    trainer = Trainer(config, opts)
    trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)  
    trainer.to(device)
    
    state_dict = torch.load(opts.pretrained_model_path)#os.path.join(opts.log_path, opts.config + '/checkpoint.pth'))
    trainer.enc.load_state_dict(torch.load(opts.pretrained_model_path))
    trainer.enc.eval()
    
    with torch.no_grad():
        img_list = [glob.glob1(align_dir_path, ext) for ext in ['*jpg','*png']]
        img_list = [item for sublist in img_list for item in sublist]
        img_list.sort()
        n_1 = trainer.StyleGAN.make_noise()
        
        for i, img_name in enumerate(img_list):
            #print(i, img_name)
            image_A = img_to_tensor(Image.open(align_dir_path + img_name)).unsqueeze(0).to(device)
            w_0, f_0 = trainer.encode(image_A)
            
            w_0_np = w_0.cpu().numpy().reshape(1, -1)
            out = linear_interpolate(w_0_np, boundary, start_distance=-step_scale, end_distance=step_scale, steps=n_steps)
            w_1 = torch.tensor(out[-1]).view(1, -1, 512).to(device)
            
            _, fea_0 = trainer.StyleGAN([w_0], noise=n_1, input_is_latent=True, return_features=True)
            _, fea_1 = trainer.StyleGAN([w_1], noise=n_1, input_is_latent=True, return_features=True)
            
            features = [None]*5 + [f_0 + fea_1[5] - fea_0[5]] + [None]*(17-5)
            x_1, _ = trainer.StyleGAN([w_1], noise=n_1, input_is_latent=True, features_in=features, feature_scale=1.0)
            utils.save_image(clip_img(x_1), process_dir_path + 'frame%04d'%i+'.jpg')
            

video_path = opts.video_path
video_name = video_path.split('/')[-1]
orig_dir_path = opts.output_path + video_name.split('.')[0] + '/' + video_name.split('.')[0] + '/'
align_dir_path = os.path.dirname(orig_dir_path) + '_crop_align/'
mask_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_mask/'
latent_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_latent/'
process_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_' + opts.attr.replace(',','_') + '/'
reproject_dir_path = os.path.dirname(orig_dir_path) + '_crop_align_' + opts.attr.replace(',','_') + '_reproject/'


print(opts.function)
start_time = time.perf_counter()

if opts.function == 'video_to_frames':
    video_to_frames(video_path, orig_dir_path, count_num=120, resize=opts.resize)
    create_video(orig_dir_path)
elif opts.function == 'align_frames':
    align_frames(orig_dir_path, align_dir_path, output_size=1024, optical_flow=opts.optical_flow, filter_size=opts.filter_size)
    # parsing mask
    parsing_net = BiSeNet(n_classes=19)
    parsing_net.load_state_dict(torch.load(opts.parsing_model_path))
    parsing_net.eval()
    parsing_net.to(device)
    generate_mask(align_dir_path, mask_dir_path, parsing_net)
elif opts.function == 'latent_manipulation':
    latent_manipulation(opts, align_dir_path, process_dir_path)
elif opts.function == 'reproject_origin':
    process_dir_path = os.path.dirname(orig_dir_path) + '_inversion/'
    reproject_dir_path = os.path.dirname(orig_dir_path) + '_inversion_reproject/'
    video_reproject(orig_dir_path, process_dir_path, reproject_dir_path, align_dir_path, mask_dir_path, seamless=opts.seamless)
    create_video(reproject_dir_path)
elif opts.function == 'reproject_manipulate':
    video_reproject(orig_dir_path, process_dir_path, reproject_dir_path, align_dir_path, mask_dir_path, seamless=opts.seamless)
    create_video(reproject_dir_path)
elif opts.function == 'compare_frames':
    process_dir_paths = []
    process_dir_paths.append(os.path.dirname(orig_dir_path) + '_inversion_reproject/')
    if len(opts.attr.split(','))>0:
        process_dir_paths.append(reproject_dir_path)
    save_dir = os.path.dirname(orig_dir_path) + '_crop_align_' + opts.attr.replace(',','_') + '_compare/'
    compare_frames(save_dir, orig_dir_path, process_dir_paths, strs=opts.strs, dim=1)
    create_video(save_dir, video_format='.avi', resize_ratio=1)
    

count_time = time.perf_counter() - start_time
print("Elapsed time: %0.4f seconds"%count_time)