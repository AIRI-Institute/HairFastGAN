# -*- coding: utf-8 -*-

"""
# File name:    model.py
# Time :        2021/11/17 15:37
# Author:       xyguoo@163.com
# Description:  
"""

import torch.nn as nn
from my_torchlib.module import LinearBlock
import torch
from torch.nn import init


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


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg

        input_dim = cfg.noise_dim
        if cfg.lambda_rgb:
            input_dim += 3
        if cfg.lambda_pca_std:
            input_dim += 1
        if cfg.lambda_cls_curliness:
            input_dim += cfg.curliness_dim

        layers = [LinearBlock(input_dim, cfg.g_hidden_dim, cfg.g_norm, activation=cfg.g_activ)]
        for _ in range(cfg.g_hidden_layer_num - 1):
            layers.append(LinearBlock(cfg.g_hidden_dim, cfg.g_hidden_dim, cfg.g_norm, activation=cfg.g_activ))
        layers.append(LinearBlock(cfg.g_hidden_dim, cfg.SEAN_code, 'none', 'none'))
        self.net = nn.Sequential(*layers)

    def forward(self, data):
        x = data['noise']
        if self.cfg.lambda_cls_curliness:
            x = torch.cat([x, data['noise_curliness']], dim=1)
        if self.cfg.lambda_rgb:
            x = torch.cat([x, data['rgb_mean']], dim=1)
        if self.cfg.lambda_pca_std:
            x = torch.cat([x, data['pca_std']], dim=1)
        output = self.net(x)
        res = {'code': output}
        return res


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        layers = [LinearBlock(cfg.SEAN_code, cfg.d_hidden_dim, cfg.d_norm, activation=cfg.d_activ)]
        for _ in range(cfg.d_hidden_layer_num - 1):
            layers.append(LinearBlock(cfg.d_hidden_dim, cfg.d_hidden_dim, cfg.d_norm, activation=cfg.d_activ))
        output_dim = 1 + cfg.noise_dim
        if cfg.lambda_rgb and 'curliness' not in cfg.predictor:
            output_dim += 3
        if cfg.lambda_pca_std:
            output_dim += 1
        if cfg.lambda_cls_curliness:
            output_dim += cfg.curliness_dim
            if 'curliness' not in cfg.predictor:
                output_dim += 1
        layers.append(LinearBlock(cfg.d_hidden_dim, output_dim, 'none', 'none'))
        self.net = nn.Sequential(*layers)
        self.input_name = 'code'

    def forward(self, data_in):
        x = data_in[self.input_name]
        out = self.net(x)
        data = {'adv': out[:, [0]]}
        ptr = 1
        data['noise'] = out[:, ptr:(ptr + self.cfg.noise_dim)]
        ptr += self.cfg.noise_dim
        if self.cfg.lambda_cls_curliness:
            data['noise_curliness'] = out[:, ptr:(ptr + self.cfg.curliness_dim)]
            ptr += self.cfg.curliness_dim
            if not 'curliness' in self.cfg.predictor:
                data['cls_curliness'] = out[:, ptr: ptr + 1]
                ptr += 1
        if self.cfg.lambda_rgb and 'rgb' not in self.cfg.predictor:
            data['rgb_mean'] = out[:, ptr:ptr + 3]
            ptr += 3
        if self.cfg.lambda_pca_std and 'rgb' not in self.cfg.predictor:
            data['pca_std'] = out[:, ptr:]
            ptr += 1
        return data

    def forward_adv_direct(self, x):
        return self.net(x)[:, [0]]


class DiscriminatorNoise(nn.Module):
    """Discriminator network."""

    def __init__(self, cfg):
        super(DiscriminatorNoise, self).__init__()
        self.cfg = cfg
        input_dim = cfg.noise_dim
        if cfg.lambda_cls_curliness:
            input_dim += cfg.curliness_dim
        layers = [LinearBlock(input_dim, cfg.d_hidden_dim, cfg.d_norm, activation=cfg.d_activ)]
        for _ in range(cfg.d_noise_hidden_layer_num - 1):
            layers.append(LinearBlock(cfg.d_hidden_dim, cfg.d_hidden_dim, cfg.d_norm, activation=cfg.d_activ))
        output_dim = 1
        layers.append(LinearBlock(cfg.d_hidden_dim, output_dim, 'none', 'none'))
        self.net = nn.Sequential(*layers)
        self.input_name = 'noise'

    def forward(self, data_in):
        x = data_in[self.input_name]
        if self.cfg.lambda_cls_curliness:
            x = torch.cat([x, data_in['noise_curliness']], dim=1)
        out = self.net(x)
        data = {'adv': out[:, [0]]}
        return data

    def forward_adv_direct(self, x):
        return self.net(x)[:, [0]]
