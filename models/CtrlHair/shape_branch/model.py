# -*- coding: utf-8 -*-

"""
# File name:    model.py
# Time :        2021/11/17 15:37
# Author:       xyguoo@163.com
# Description:  
"""

import torch.nn as nn
from models.CtrlHair.my_torchlib.module import LinearBlock, Conv2dBlock
import torch
from torch.nn import init
import numpy as np
from models.CtrlHair.global_value_utils import HAIR_IDX


def generate_pos_embedding(img_size, order=10):
    coordinators = np.linspace(0, 1, img_size, endpoint=False)
    bi_coordinators = np.stack(np.meshgrid(coordinators, coordinators), 0)
    bi_coordinators = bi_coordinators[None, ...]
    nums = np.arange(0, order)
    nums = 2 ** nums * np.pi
    nums = nums[:, None, None, None]
    gamma1 = np.sin(nums * bi_coordinators)
    gamma2 = np.cos(nums * bi_coordinators)
    gamma = np.concatenate([gamma1, gamma2], axis=0)
    gamma = gamma.reshape([-1, img_size, img_size])
    gamma = torch.tensor(gamma, requires_grad=False).float()
    return gamma


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class MaskEncoder(nn.Module):

    def __init__(self, input_channel, output_dim, norm, layer_num, input_size=256, vae_mode=False,
                 pos_encoding_order=10, max_batch_size=1, hidden_in_channel=32):
        super(MaskEncoder, self).__init__()
        self.vae_mode = vae_mode
        layers = []

        in_channel = input_channel + pos_encoding_order * 4
        for cur_num in range(layer_num):
            out_channel = min(2048, 2 ** cur_num * hidden_in_channel)
            cur_conv = Conv2dBlock(
                in_channel, out_channel, kernel_size=4, stride=2, padding=1, norm=norm, activation='lrelu')
            layers.append(cur_conv)
            in_channel = out_channel

        out_size = input_size // (2 ** layer_num)
        self.layers = nn.Sequential(*layers)

        fc_in_dim = out_size ** 2 * out_channel
        self.out_layer = LinearBlock(fc_in_dim, output_dim, norm='none', activation='none')
        if self.vae_mode:
            self.std_out_layer = LinearBlock(fc_in_dim, output_dim, norm='none', activation='none')

        self.input_embedding = generate_pos_embedding(img_size=input_size, order=pos_encoding_order)
        self.input_embedding = self.input_embedding.repeat((max_batch_size, 1, 1, 1))

    def forward(self, input_mask: torch.Tensor):
        batch_size = input_mask.shape[0]
        if self.input_embedding.device != input_mask.device:
            self.input_embedding = self.input_embedding.to(input_mask.device)
        input_with_pos = torch.cat([input_mask, self.input_embedding[:batch_size]], axis=1)
        feature = self.layers(input_with_pos)
        feature = feature.flatten(1)
        out_mean = self.out_layer(feature)
        if self.vae_mode:
            out_std = self.std_out_layer(feature).abs()
            return self.vae_resampling(out_mean, out_std), out_mean, out_std
        else:
            return out_mean, out_mean, None

    def vae_resampling(self, mean, std):
        z = torch.randn(mean.shape).to(mean.device)
        res = z * std + mean
        return res


class MaskDecoder(nn.Module):
    def __init__(self, input_dim, output_channel, norm, layer_num, output_size=256):
        super(MaskDecoder, self).__init__()

        self.in_channel = min(32 * 2 ** layer_num, 2048)
        self.input_size = output_size // (2 ** layer_num)
        self.in_layer = LinearBlock(input_dim, self.in_channel * self.input_size ** 2, norm='none', activation='none')

        layers = []
        in_channel = self.in_channel
        for cur_num in range(layer_num):
            up = nn.Upsample(scale_factor=2, mode='nearest')
            out_channel = min(32 * 2 ** (layer_num - 1 - cur_num), 2048)
            cur_conv = Conv2dBlock(in_channel, out_channel, kernel_size=3, stride=1, padding=1, norm=norm,
                                   activation='lrelu')
            layers.append(up)
            layers.append(cur_conv)
            in_channel = out_channel
        self.layers = nn.Sequential(*layers)
        self.out_layer = Conv2dBlock(in_channel, output_channel, kernel_size=3, stride=1, padding=1, norm='none',
                                     activation='none')

    def forward(self, input_vector):
        feature = self.in_layer(input_vector)
        feature = feature.reshape(-1, self.in_channel, self.input_size, self.input_size)
        feature = self.layers(feature)
        feature = self.out_layer(feature)
        return feature


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg

        self.hair_encoder = MaskEncoder(1, cfg.hair_dim, norm=cfg.g_norm, layer_num=7, vae_mode=cfg.vae_hair_mode,
        # self.hair_encoder = MaskEncoder(1, cfg.hair_dim, norm='none', layer_num=7, vae_mode=cfg.vae_hair_mode,
                                        pos_encoding_order=cfg.pos_encoding_order,
                                        max_batch_size=max(cfg.total_batch_size, cfg.sample_batch_size))
        self.face_encoder = MaskEncoder(18, 1024, norm=cfg.g_norm, layer_num=7, vae_mode=False,
        # self.face_encoder = MaskEncoder(18, 1024, norm='none', layer_num=7, vae_mode=False,
                                        pos_encoding_order=cfg.pos_encoding_order,
                                        max_batch_size=max(cfg.total_batch_size, cfg.sample_batch_size))
        self.hair_decoder = MaskDecoder(1024 + cfg.hair_dim, output_channel=1, norm=cfg.g_norm, layer_num=7)
        self.face_decoder = MaskDecoder(1024, output_channel=18, norm=cfg.g_norm, layer_num=7)

    def forward_hair_encoder(self, hair, testing=False):
        code, mean, std = self.hair_encoder(hair)
        if testing:
            return mean
        else:
            return code, mean, std

    def forward_face_encoder(self, face):
        code, _, _ = self.face_encoder(face)
        return code

    def forward_hair_decoder(self, hair_code, face_code):
        code = torch.cat([face_code, hair_code], dim=1)
        hair = self.hair_decoder(code)
        return hair

    def forward_face_decoder(self, face_code):
        face = self.face_decoder(face_code)
        return face

    def forward_decoder(self, hair_logit, face_logit):
        logit = torch.cat([face_logit[:, :HAIR_IDX], hair_logit, face_logit[:, HAIR_IDX:]], dim=1)
        mask = torch.softmax(logit, dim=1)
        return mask

    def forward_edit_directly_in_test(self, hair, face):
        _, hair_code, _ = self.forward_hair_encoder(hair)
        face_code = self.forward_face_encoder(face)
        mask = self.forward_decode_by_code(hair_code, face_code)
        return mask

    def forward_decode_by_code(self, hair_code, face_code):
        hair_logit = self.forward_hair_decoder(hair_code, face_code)
        face_logit = self.forward_face_decoder(face_code)
        mask = self.forward_decoder(hair_logit, face_logit)
        return mask


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        self.dis = MaskEncoder(19, 1, norm=cfg.d_norm, layer_num=7, vae_mode=False,
                               pos_encoding_order=cfg.pos_encoding_order, max_batch_size=cfg.total_batch_size,
                               hidden_in_channel=cfg.d_hidden_in_channel)

    def forward(self, mask):
        dis_res, _, _ = self.dis(mask)
        return dis_res


class DiscriminatorNoise(nn.Module):
    """Discriminator network."""

    def __init__(self, cfg):
        super(DiscriminatorNoise, self).__init__()
        self.cfg = cfg
        input_dim = cfg.hair_dim
        layers = [LinearBlock(input_dim, cfg.d_hidden_dim, cfg.d_norm, activation='lrelu')]
        for _ in range(cfg.d_noise_hidden_layer_num - 1):
            layers.append(LinearBlock(cfg.d_hidden_dim, cfg.d_hidden_dim, cfg.d_norm, activation='lrelu'))
        output_dim = 1
        layers.append(LinearBlock(cfg.d_hidden_dim, output_dim, 'none', 'none'))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)[:, [0]]
