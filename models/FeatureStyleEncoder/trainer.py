import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torch.autograd import grad
from torchvision import transforms, utils

import face_alignment
import lpips

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)
from pixel2style2pixel.models.stylegan2.model import Generator, get_keys

from nets.feature_style_encoder import *
from arcface.iresnet import *
from face_parsing.model import BiSeNet
from ranger import Ranger

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torch.autograd import grad

        
def clip_img(x):
    """Clip stylegan generated image to range(0,1)"""
    img_tmp = x.clone()[0]
    img_tmp = (img_tmp + 1) / 2
    img_tmp = torch.clamp(img_tmp, 0, 1)
    return [img_tmp.detach().cpu()]

def tensor_byte(x):
    return x.element_size()*x.nelement()

def count_parameters(net):
    s = sum([np.prod(list(mm.size())) for mm in net.parameters()])
    print(s)

def stylegan_to_classifier(x, out_size=(224, 224)):
    """Clip image to range(0,1)"""
    img_tmp = x.clone()
    img_tmp = torch.clamp((0.5*img_tmp + 0.5), 0, 1)
    img_tmp = F.interpolate(img_tmp, size=out_size, mode='bilinear')
    img_tmp[:,0] = (img_tmp[:,0] - 0.485)/0.229
    img_tmp[:,1] = (img_tmp[:,1] - 0.456)/0.224
    img_tmp[:,2] = (img_tmp[:,2] - 0.406)/0.225
    #img_tmp = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tmp)
    return img_tmp
    
def downscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode=mode)
    return x
    
def upscale(x, scale_times=1, mode='bilinear'):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=2, mode=mode)
    return x
    
def hist_transform(source_tensor, target_tensor):
    """Histogram transformation"""
    c, h, w = source_tensor.size()
    s_t = source_tensor.view(c, -1)
    t_t = target_tensor.view(c, -1)
    s_t_sorted, s_t_indices = torch.sort(s_t)
    t_t_sorted, t_t_indices = torch.sort(t_t)
    for i in range(c):
        s_t[i, s_t_indices[i]] = t_t_sorted[i]
    return s_t.view(c, h, w)

def init_weights(m):
    """Initialize layers with Xavier uniform distribution"""
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        nn.init.uniform_(m.weight, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

def total_variation(x, delta=1):
    """Total variation, x: tensor of size (B, C, H, W)"""
    out = torch.mean(torch.abs(x[:, :, :, :-delta] - x[:, :, :, delta:]))\
        + torch.mean(torch.abs(x[:, :, :-delta, :] - x[:, :, delta:, :]))
    return out

def vgg_transform(x):
    """Adapt image for vgg network, x: image of range(0,1) subtracting ImageNet mean"""
    r, g, b = torch.split(x, 1, 1)
    out = torch.cat((b, g, r), dim = 1)
    out = F.interpolate(out, size=(224, 224), mode='bilinear')
    out = out*255.
    return out

# warp image with flow
def normalize_axis(x,L):
    return (x-1-(L-1)/2)*2/(L-1)

def unnormalize_axis(x,L):
    return x*(L-1)/2+1+(L-1)/2

def torch_flow_to_th_sampling_grid(flow,h_src,w_src,use_cuda=False):
    b,c,h_tgt,w_tgt=flow.size()
    grid_y, grid_x = torch.meshgrid(torch.tensor(range(1,w_tgt+1)),torch.tensor(range(1,h_tgt+1)))
    disp_x=flow[:,0,:,:]
    disp_y=flow[:,1,:,:]
    source_x=grid_x.unsqueeze(0).repeat(b,1,1).type_as(flow)+disp_x
    source_y=grid_y.unsqueeze(0).repeat(b,1,1).type_as(flow)+disp_y
    source_x_norm=normalize_axis(source_x,w_src) 
    source_y_norm=normalize_axis(source_y,h_src) 
    sampling_grid=torch.cat((source_x_norm.unsqueeze(3), source_y_norm.unsqueeze(3)), dim=3)
    if use_cuda:
        sampling_grid = sampling_grid.cuda()
    return sampling_grid

def warp_image_torch(image, flow):
    """
    Warp image (tensor, shape=[b, 3, h_src, w_src]) with flow (tensor, shape=[b, h_tgt, w_tgt, 2])
    """
    b,c,h_src,w_src=image.size()
    sampling_grid_torch = torch_flow_to_th_sampling_grid(flow, h_src, w_src)  
    warped_image_torch = F.grid_sample(image, sampling_grid_torch)
    return warped_image_torch

class Trainer(nn.Module):
    def __init__(self, config, opts):
        super(Trainer, self).__init__()
        # Load Hyperparameters
        self.config = config
        self.device = torch.device(self.config['device'])
        self.scale = int(np.log2(config['resolution']/config['enc_resolution']))
        self.scale_mode = 'bilinear'
        self.opts = opts
        self.n_styles = 2 * int(np.log2(config['resolution'])) - 2
        self.idx_k = 5
        if 'idx_k' in self.config:
            self.idx_k = self.config['idx_k']
        if 'stylegan_version' in self.config and self.config['stylegan_version'] == 3:
            self.n_styles = 16
        # Networks
        in_channels = 256
        if 'in_c' in self.config:
            in_channels = config['in_c']
        enc_residual = False
        if 'enc_residual' in self.config:
            enc_residual = self.config['enc_residual']
        enc_residual_coeff = False
        if 'enc_residual_coeff' in self.config:
            enc_residual_coeff = self.config['enc_residual_coeff']
        resnet_layers = [4,5,6]
        if 'enc_start_layer' in self.config:
            st_l = self.config['enc_start_layer']
            resnet_layers = [st_l, st_l+1, st_l+2]
        if 'scale_mode' in self.config:
            self.scale_mode = self.config['scale_mode']
        # Load encoder
        self.stride = (self.config['fs_stride'], self.config['fs_stride'])
        self.enc = fs_encoder_v2(n_styles=self.n_styles, opts=opts, residual=enc_residual, use_coeff=enc_residual_coeff, resnet_layer=resnet_layers, stride=self.stride)
        
        ##########################
        # Other nets
        self.StyleGAN = self.init_stylegan(config)
        self.Arcface = iresnet50()
        self.parsing_net = BiSeNet(n_classes=19)
        # Optimizers
        # Latent encoder
        self.enc_params = list(self.enc.parameters()) 
        if 'freeze_iresnet' in self.config and self.config['freeze_iresnet']:
            self.enc_params =  list(self.enc.styles.parameters())
        if 'optimizer' in self.config and self.config['optimizer'] == 'ranger':
            self.enc_opt = Ranger(self.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        else:
            self.enc_opt = torch.optim.Adam(self.enc_params, lr=config['lr'], betas=(config['beta_1'], config['beta_2']), weight_decay=config['weight_decay'])
        self.enc_scheduler = torch.optim.lr_scheduler.StepLR(self.enc_opt, step_size=config['step_size'], gamma=config['gamma'])

        self.fea_avg = None

    def initialize(self, stylegan_model_path, arcface_model_path, parsing_model_path):
        # load StyleGAN model
        stylegan_state_dict = torch.load(stylegan_model_path, map_location='cpu')
        self.StyleGAN.load_state_dict(get_keys(stylegan_state_dict, 'decoder'), strict=True)
        self.StyleGAN.to(self.device)
        # get StyleGAN average latent in w space and the noise inputs
        self.dlatent_avg = stylegan_state_dict['latent_avg'].to(self.device)
        self.noise_inputs = [getattr(self.StyleGAN.noises, f'noise_{i}').to(self.device) for i in range(self.StyleGAN.num_layers)]
        # load Arcface weight
        self.Arcface.load_state_dict(torch.load(self.opts.arcface_model_path))
        self.Arcface.eval()
        # load face parsing net weight
        self.parsing_net.load_state_dict(torch.load(self.opts.parsing_model_path))
        self.parsing_net.eval()
        # load lpips net weight
        # self.loss_fn = lpips.LPIPS(net='alex', spatial=False)
        # self.loss_fn.to(self.device)
    
    def init_stylegan(self, config):
        """StyleGAN = G_main(
            truncation_psi=config['truncation_psi'], 
            resolution=config['resolution'], 
            use_noise=config['use_noise'],  
            randomize_noise=config['randomize_noise']
        )"""
        StyleGAN = Generator(1024, 512, 8)
        return StyleGAN
    
    def mapping(self, z):
        return self.StyleGAN.get_latent(z).detach()

    def L1loss(self, input, target):
        return nn.L1Loss()(input,target)
    
    def L2loss(self, input, target):
        return nn.MSELoss()(input,target)

    def CEloss(self, x, target_age):
        return nn.CrossEntropyLoss()(x, target_age)
    
    def LPIPS(self, input, target, multi_scale=False):
        if multi_scale:
            out = 0
            for k in range(3):
                out += self.loss_fn.forward(downscale(input, k, self.scale_mode), downscale(target, k, self.scale_mode)).mean()
        else:
            out = self.loss_fn.forward(downscale(input, self.scale, self.scale_mode), downscale(target, self.scale, self.scale_mode)).mean()
        return out
    
    def IDloss(self, input, target):
        x_1 = F.interpolate(input, (112,112))
        x_2 = F.interpolate(target, (112,112))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if 'multi_layer_idloss' in self.config and self.config['multi_layer_idloss']:
            id_1 = self.Arcface(x_1, return_features=True)
            id_2 = self.Arcface(x_2, return_features=True)
            return sum([1 - cos(id_1[i].flatten(start_dim=1), id_2[i].flatten(start_dim=1)) for i in range(len(id_1))])
        else:
            id_1 = self.Arcface(x_1)
            id_2 = self.Arcface(x_2)
            return 1 - cos(id_1, id_2)
    
    def landmarkloss(self, input, target):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        x_1 = stylegan_to_classifier(input, out_size=(512, 512))
        x_2 = stylegan_to_classifier(target, out_size=(512,512))
        out_1 = self.parsing_net(x_1)
        out_2 = self.parsing_net(x_2)
        parsing_loss = sum([1 - cos(out_1[i].flatten(start_dim=1), out_2[i].flatten(start_dim=1)) for i in range(len(out_1))])
        return parsing_loss.mean()
        

    def feature_match(self, enc_feat, dec_feat, layer_idx=None):
        loss = []
        if layer_idx is None:
            layer_idx = [i for i in range(len(enc_feat))]
        for i in layer_idx:
            loss.append(self.L1loss(enc_feat[i], dec_feat[i]))
        return loss
    
    def encode(self, img):
        w_recon, fea = self.enc(downscale(img, self.scale, self.scale_mode)) 
        w_recon = w_recon + self.dlatent_avg
        return w_recon, fea

    def get_image(self, w=None, img=None, noise=None, zero_noise_input=True, training_mode=True):
        
        x_1, n_1 = img, noise
        if x_1 is None:
            x_1, _ = self.StyleGAN([w], input_is_latent=True, noise = n_1)
           
        w_delta = None
        fea = None
        features = None
        return_features = False
        # Reconstruction
        k = 0
        if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
            return_features = True
            k = self.idx_k
            w_recon, fea = self.enc(downscale(x_1, self.scale, self.scale_mode)) 
            w_recon = w_recon + self.dlatent_avg
            features = [None]*k + [fea] + [None]*(17-k)
        else:
            w_recon = self.enc(downscale(x_1, self.scale, self.scale_mode)) + self.dlatent_avg        

        # generate image
        x_1_recon, fea_recon = self.StyleGAN([w_recon], input_is_latent=True, return_features=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter))
        fea_recon = fea_recon[k].detach() 
        return [x_1_recon, x_1[:,:3,:,:], w_recon, w_delta, n_1, fea, fea_recon]

    def compute_loss(self, w=None, img=None, noise=None, real_img=None):
        return self.compute_loss_stylegan2(w=w, img=img, noise=noise, real_img=real_img)

    def compute_loss_stylegan2(self, w=None, img=None, noise=None, real_img=None):
        
        if img is None:
            # generate synthetic images
            if noise is None:
                noise = [torch.randn(w.size()[:1] + ee.size()[1:]).to(self.device) for ee in self.noise_inputs]
            img, _ = self.StyleGAN([w], input_is_latent=True, noise = noise)
            img = img.detach()

        if img is not None and real_img is not None:
            # concat synthetic and real data
            img = torch.cat([img, real_img], dim=0)
            noise = [torch.cat([ee, ee], dim=0) for ee in noise]
        
        out = self.get_image(w=w, img=img, noise=noise)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1, fea_recon = out

        # Loss setting
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w']['id']
        b = x_1.size(0)//2
        if 'l2loss_on_real_image' in self.config and self.config['l2loss_on_real_image']:
            b = x_1.size(0)
        self.l2_loss = self.L2loss(x_1_recon[:b], x_1[:b]) if w_l2 > 0 else torch.tensor(0) # l2 loss only on synthetic data
        # LPIPS
        multiscale_lpips=False if 'multiscale_lpips' not in self.config else self.config['multiscale_lpips']
        self.lpips_loss = self.LPIPS(x_1_recon, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
        self.id_loss = self.IDloss(x_1_recon, x_1).mean() if w_id > 0 else torch.tensor(0)
        self.landmark_loss = self.landmarkloss(x_1_recon, x_1) if self.config['w']['landmark'] > 0 else torch.tensor(0)
        
        if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
            k = self.idx_k 
            features = [None]*k + [fea_1] + [None]*(17-k)
            x_1_recon_2, _ = self.StyleGAN([w_recon], noise=n_1, input_is_latent=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter))
            self.lpips_loss += self.LPIPS(x_1_recon_2, x_1, multi_scale=multiscale_lpips).mean() if w_lpips > 0 else torch.tensor(0)
            self.id_loss += self.IDloss(x_1_recon_2, x_1).mean() if w_id > 0 else torch.tensor(0)
            self.landmark_loss += self.landmarkloss(x_1_recon_2, x_1) if self.config['w']['landmark'] > 0 else torch.tensor(0)

        # downscale image
        x_1 = downscale(x_1, self.scale, self.scale_mode)
        x_1_recon = downscale(x_1_recon, self.scale, self.scale_mode)
        
        # Total loss
        w_l2, w_lpips, w_id = self.config['w']['l2'], self.config['w']['lpips'], self.config['w']['id']
        self.loss = w_l2*self.l2_loss + w_lpips*self.lpips_loss + w_id*self.id_loss
        
        if 'f_recon' in self.config['w']:
            self.feature_recon_loss = self.L2loss(fea_1, fea_recon) 
            self.loss += self.config['w']['f_recon']*self.feature_recon_loss
        if 'l1' in self.config['w'] and self.config['w']['l1']>0:
            self.l1_loss = self.L1loss(x_1_recon, x_1)
            self.loss += self.config['w']['l1']*self.l1_loss
        if 'landmark' in self.config['w']:
            self.loss += self.config['w']['landmark']*self.landmark_loss
        return self.loss

    def test(self, w=None, img=None, noise=None, zero_noise_input=True, return_latent=False, training_mode=False):        
        if 'n_iter' not in self.__dict__.keys():
            self.n_iter = 1e5
        out = self.get_image(w=w, img=img, noise=noise, training_mode=training_mode)
        x_1_recon, x_1, w_recon, w_delta, n_1, fea_1 = out[:6]
        output = [x_1, x_1_recon]
        if return_latent:
            output += [w_recon, fea_1]
        return output

    def log_loss(self, logger, n_iter, prefix='scripts'):
        logger.log_value(prefix + '/l2_loss', self.l2_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/lpips_loss', self.lpips_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/id_loss', self.id_loss.item(), n_iter + 1)
        logger.log_value(prefix + '/total_loss', self.loss.item(), n_iter + 1)
        if 'f_recon' in self.config['w']:
            logger.log_value(prefix + '/feature_recon_loss', self.feature_recon_loss.item(), n_iter + 1)
        if 'l1' in self.config['w'] and self.config['w']['l1']>0:
            logger.log_value(prefix + '/l1_loss', self.l1_loss.item(), n_iter + 1)
        if 'landmark' in self.config['w']:
            logger.log_value(prefix + '/landmark_loss', self.landmark_loss.item(), n_iter + 1)
        
    def save_image(self, log_dir, n_epoch, n_iter, prefix='/scripts/', w=None, img=None, noise=None, training_mode=True):
        return self.save_image_stylegan2(log_dir=log_dir, n_epoch=n_epoch, n_iter=n_iter, prefix=prefix, w=w, img=img, noise=noise, training_mode=training_mode)

    def save_image_stylegan2(self, log_dir, n_epoch, n_iter, prefix='/scripts/', w=None, img=None, noise=None, training_mode=True):
        os.makedirs(log_dir + prefix, exist_ok=True)
        with torch.no_grad():
            out = self.get_image(w=w, img=img, noise=noise, training_mode=training_mode)
            x_1_recon, x_1, w_recon, w_delta, n_1, fea_1 = out[:6]
            x_1 = downscale(x_1, self.scale, self.scale_mode)
            x_1_recon = downscale(x_1_recon, self.scale, self.scale_mode)
            out_img = torch.cat((x_1, x_1_recon), dim=3)
            #fs
            if 'use_fs_encoder' in self.config and self.config['use_fs_encoder']:
                k = self.idx_k 
                features = [None]*k + [fea_1] + [None]*(17-k)
                x_1_recon_2, _ = self.StyleGAN([w_recon], noise=n_1, input_is_latent=True, features_in=features, feature_scale=min(1.0, 0.0001*self.n_iter))
                x_1_recon_2 = downscale(x_1_recon_2, self.scale, self.scale_mode)
                out_img = torch.cat((x_1, x_1_recon, x_1_recon_2), dim=3)
            utils.save_image(clip_img(out_img[:1]), log_dir + prefix + 'epoch_' +str(n_epoch+1) + '_iter_' + str(n_iter+1) + '_0.jpg')
            if out_img.size(0)>1:
                utils.save_image(clip_img(out_img[1:]), log_dir + prefix + 'epoch_' +str(n_epoch+1) + '_iter_' + str(n_iter+1) + '_1.jpg')
                        
    def save_model(self, log_dir):
        torch.save(self.enc.state_dict(),'{:s}/enc.pth.tar'.format(log_dir))

    def save_checkpoint(self, n_epoch, log_dir):
        checkpoint_state = {
            'n_epoch': n_epoch,
            'enc_state_dict': self.enc.state_dict(),
            'enc_opt_state_dict': self.enc_opt.state_dict(),
            'enc_scheduler_state_dict': self.enc_scheduler.state_dict()
        }
        torch.save(checkpoint_state, '{:s}/checkpoint.pth'.format(log_dir))
        if (n_epoch+1)%10 == 0 :
            torch.save(checkpoint_state, '{:s}/checkpoint'.format(log_dir)+'_'+str(n_epoch+1)+'.pth')
    
    def load_model(self, log_dir):
        self.enc.load_state_dict(torch.load('{:s}/enc.pth.tar'.format(log_dir)))

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.enc.load_state_dict(state_dict['enc_state_dict'])
        self.enc_opt.load_state_dict(state_dict['enc_opt_state_dict'])
        self.enc_scheduler.load_state_dict(state_dict['enc_scheduler_state_dict'])
        return state_dict['n_epoch'] + 1

    def update(self, w=None, img=None, noise=None, real_img=None, n_iter=0):
        self.n_iter = n_iter
        self.enc_opt.zero_grad()
        self.compute_loss(w=w, img=img, noise=noise, real_img=real_img).backward()
        self.enc_opt.step()

        
