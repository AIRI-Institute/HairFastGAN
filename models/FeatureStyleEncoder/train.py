import argparse
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
from tensorboard_logger import Logger

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
parser.add_argument('--real_dataset_path', type=str, default='./data/ffhq-dataset/images/', help='dataset path')
parser.add_argument('--dataset_path', type=str, default='./data/stylegan2-generate-images/ims/', help='dataset path')
parser.add_argument('--label_path', type=str, default='./data/stylegan2-generate-images/seeds_pytorch_1.8.1.npy', help='laebl path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='pretrained stylegan2 model')
parser.add_argument('--arcface_model_path', type=str, default='./pretrained_models/backbone.pth', help='pretrained ArcFace model')
parser.add_argument('--parsing_model_path', type=str, default='./pretrained_models/79999_iter.pth', help='pretrained parsing model')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
opts = parser.parse_args()

log_dir = os.path.join(opts.log_path, opts.config) + '/'
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_dir)

config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)

batch_size = config['batch_size']
epochs = config['epochs']
iter_per_epoch = config['iter_per_epoch']
img_size = (config['resolution'], config['resolution'])
video_data_input = False


img_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_to_tensor_car = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.Pad(padding=(0, 64, 0, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Initialize trainer
trainer = Trainer(config, opts)
trainer.initialize(opts.stylegan_model_path, opts.arcface_model_path, opts.parsing_model_path)   
trainer.to(device)

noise_exemple = trainer.noise_inputs
train_data_split = 0.9 if 'train_split' not in config else config['train_split']

# Load synthetic dataset
dataset_A = MyDataSet(image_dir=opts.dataset_path, label_dir=opts.label_path, output_size=img_size, noise_in=noise_exemple, training_set=True, train_split=train_data_split)
loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# Load real dataset
dataset_B = MyDataSet(image_dir=opts.real_dataset_path, label_dir=None, output_size=img_size, noise_in=noise_exemple, training_set=True, train_split=train_data_split)
loader_B = data.DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Start Training
epoch_0 = 0

# check if checkpoint exist
if 'checkpoint.pth' in os.listdir(log_dir):
    epoch_0 = trainer.load_checkpoint(os.path.join(log_dir, 'checkpoint.pth'))

if opts.resume:
    epoch_0 = trainer.load_checkpoint(os.path.join(opts.log_path, opts.checkpoint))

torch.manual_seed(0)
os.makedirs(log_dir + 'validation/', exist_ok=True)

print("Start!")

for n_epoch in tqdm(range(epoch_0, epochs)):

    iter_A = iter(loader_A)
    iter_B = iter(loader_B)
    iter_0 = n_epoch*iter_per_epoch

    trainer.enc_opt.zero_grad()

    for n_iter in range(iter_0, iter_0 + iter_per_epoch):
        
        if opts.dataset_path is None:
            z, noise = next(iter_A)
            img_A = None
        else:
            z, img_A, noise = next(iter_A)
            img_A = img_A.to(device)

        z = z.to(device)
        noise = [ee.to(device) for ee in noise]
        w = trainer.mapping(z)
        if 'fixed_noise' in config and config['fixed_noise']:
            img_A, noise = None, None

        img_B = None
        if 'use_realimg' in config and config['use_realimg']:
            try:
                img_B = next(iter_B)
                if img_B.size(0) != batch_size:
                    iter_B = iter(loader_B)
                    img_B = next(iter_B)
            except StopIteration:
                iter_B = iter(loader_B)
                img_B = next(iter_B)
            img_B = img_B.to(device)
            
        trainer.update(w=w, img=img_A, noise=noise, real_img=img_B, n_iter=n_iter)
        if (n_iter+1) % config['log_iter'] == 0:
            trainer.log_loss(logger, n_iter, prefix='scripts')
        if (n_iter+1) % config['image_save_iter'] == 0:
            trainer.save_image(log_dir, n_epoch, n_iter, prefix='/scripts/', w=w, img=img_A, noise=noise)
            trainer.save_image(log_dir, n_epoch, n_iter+1, prefix='/scripts/', w=w, img=img_B, noise=noise, training_mode=False)
            
    trainer.enc_scheduler.step()
    trainer.save_checkpoint(n_epoch, log_dir)
    
    # Test the model on celeba hq dataset
    with torch.no_grad():
        trainer.enc.eval()
        for i in range(10):
            image_A = img_to_tensor(Image.open('./data/celeba_hq/%d.jpg' % i)).unsqueeze(0).to(device)
            output = trainer.test(img=image_A)
            out_img = torch.cat(output, 3)
            utils.save_image(clip_img(out_img[:1]), log_dir + 'validation/' + 'epoch_' +str(n_epoch+1) + '_' + str(i) + '.jpg')
        trainer.compute_loss(w=w, img=img_A, noise=noise, real_img=img_B)
        trainer.log_loss(logger, n_iter, prefix='validation')

trainer.save_model(log_dir)