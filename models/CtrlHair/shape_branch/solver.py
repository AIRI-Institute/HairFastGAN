# -*- coding: utf-8 -*-

"""
# File name:    solver.py
# Time :        2021/11/17 16:24
# Author:       xyguoo@163.com
# Description:  
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP

from models.CtrlHair.my_torchlib.train_utils import generate_noise
from .config import cfg
from .model import Generator, Discriminator, DiscriminatorNoise
# solver
from .shape_util import mask_label_to_one_hot, mask_one_hot_to_label, split_hair_face


class Solver:

    def __init__(self, cfg, device, local_rank, training=True):

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        self.cfg = cfg
        self.gen = Generator(cfg)
        self.dis = Discriminator(cfg)

        self.gen.to(device)
        self.dis.to(device)

        if training:
            self.G_optimizer = torch.optim.Adam(self.gen.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2),
                                                weight_decay=0.00)
            self.D_optimizer = torch.optim.Adam(self.dis.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2),
                                                weight_decay=0.00)
            if cfg.lambda_adv_noise:
                self.dis_noise = DiscriminatorNoise(cfg)
                self.dis_noise.to(device)
                self.D_noise_optimizer = torch.optim.Adam(self.dis_noise.parameters(), lr=cfg.lr_g,
                                                          betas=(cfg.beta1, cfg.beta2), weight_decay=0.00)
            else:
                self.dis_noise = None
        else:
            self.gen.eval()
            self.dis.eval()

        if local_rank >= 0:
            pDDP = lambda m, find_unused: DDP(m, device_ids=[local_rank], output_device=local_rank,
                                              find_unused_parameters=False)
            self.gen = pDDP(self.gen, find_unused=True)
            self.dis = pDDP(self.dis, find_unused=True)
        self.local_rank = local_rank
        self.device = device

    @staticmethod
    def kl_loss(mean, std):
        var = std ** 2
        var_log = torch.log(var + 1e-4)
        kl_loss = 0.5 * (mean ** 2 + 1.0 * (var - 1 - var_log)).mean()
        return kl_loss

    def forward(self, data):
        self.data_target_mask = data['target']
        self.data_face_mask = data['face']
        self.data_hair_mask = data['hair']

        if cfg.disturb_real_batch_mask:
            data_target_disturb = self.disturb_real(data['target'])
            self.ae_in_hair, self.ae_in_target_face = split_hair_face(data_target_disturb)
            data_face_disturb = self.disturb_real(data['face'])
            _, self.ae_in_face = split_hair_face(data_face_disturb)
        else:
            self.ae_in_hair, self.ae_in_target_face = split_hair_face(data['target'])
            _, self.ae_in_face = split_hair_face(data['face'])

        self.ae_mid_hair_code, self.ae_mid_hair_mean, self.ae_mid_hair_std = self.gen.forward_hair_encoder(
            self.ae_in_hair)
        self.ae_mid_face_code = self.gen.forward_face_encoder(self.ae_in_face)
        self.ae_out_hair_logit = self.gen.forward_hair_decoder(self.ae_mid_hair_code, self.ae_mid_face_code)
        self.ae_out_face_logit = self.gen.forward_face_decoder(self.ae_mid_face_code)

        self.ae_out_mask = self.gen.forward_decoder(self.ae_out_hair_logit, self.ae_out_face_logit)

        if 'lambda_adv_noise' in cfg or 'lambda_info' in cfg:
            self.real_noise = generate_noise(cfg.batch_size, cfg.hair_dim).to(
                self.ae_mid_face_code.device).type_as(self.ae_mid_face_code)

        if 'lambda_info' in cfg:
            self.gan_in_hair_code = self.real_noise
            self.gan_in_face_code = self.ae_mid_face_code
            self.gan_mid_hair_logit = self.gen.forward_hair_decoder(self.gan_in_hair_code, self.gan_in_face_code)
            self.gan_mid_face_logit = self.gen.forward_face_decoder(self.gan_in_face_code)
            self.gan_mid_mask = self.gen.forward_decoder(self.gan_mid_hair_logit, self.gan_mid_face_logit)

            self.gan_mid_hair, _ = split_hair_face(self.gan_mid_mask)
            self.gan_out_hair_code, _, _ = self.gen.forward_hair_encoder(self.gan_mid_hair)

            if random.random() < 0.5:
                self.dis_out_fake = self.dis.forward(self.ae_out_mask)
            else:
                self.dis_out_fake = self.dis.forward(self.gan_mid_mask)
        else:
            if random.random() < cfg.random_ae_prob:
                self.dis_out_fake = self.dis.forward(self.ae_out_mask)
            else:
                self.gan_in_hair_code = self.real_noise
                self.gan_mid_hair_logit = self.gen.forward_hair_decoder(self.gan_in_hair_code,
                                                                        self.ae_mid_face_code)
                self.gan_mid_face_logit = self.ae_out_face_logit
                self.gan_mid_mask = self.gen.forward_decoder(self.gan_mid_hair_logit, self.gan_mid_face_logit)
                self.dis_out_fake = self.dis.forward(self.gan_mid_mask)

    def forward_g(self, loss_dict):
        self.forward_general_gen(self.dis_out_fake, loss_dict)
        hair, face = split_hair_face(self.ae_out_mask)

        if cfg.regular_method == 'ce':
            loss_dict['lambda_hair'] = -torch.log(hair + 1e-5)[(self.ae_in_hair > 0.5)].mean()
            loss_dict['lambda_non_hair'] = -torch.log(1 - hair + 1e-5)[(self.ae_in_hair < 0.5)].mean()
            loss_dict['lambda_face'] = -torch.log(face + 1e-5)[(self.ae_in_target_face > 0.5)].mean()

        hair_hair, hair_face = split_hair_face(self.data_hair_mask)
        mask = self.gen.forward_edit_directly_in_test(hair_hair, hair_face)
        if 'lambda_self_rec' in cfg:
            if cfg.regular_method == 'ce':
                loss_dict['lambda_self_rec'] = -torch.log(mask + 1e-5)[self.data_hair_mask > 0.5].mean()

        if 'lambda_kl' in cfg and cfg.lambda_kl > 0:
            loss_dict['lambda_kl'] = self.kl_loss(self.ae_mid_hair_mean, self.ae_mid_hair_std)
        if cfg.lambda_moment_1 or cfg.lambda_moment_2:
            noise_mid = self.ae_mid_hair_code
            if cfg.lambda_moment_1:
                loss_dict['lambda_moment_1'] = (noise_mid.mean(dim=0) ** 2).mean()
            if cfg.lambda_moment_2:
                loss_dict['lambda_moment_2'] = (((noise_mid ** 2).mean(dim=0) - 0.973) ** 2).mean()

        if 'lambda_info' in cfg:
            loss_dict['lambda_info'] = self.mse_loss(self.gan_out_hair_code, self.gan_in_hair_code)

        if cfg.lambda_adv_noise:
            self.d_noise_res = self.dis_noise(self.ae_mid_hair_code)
            self.forward_general_gen(self.d_noise_res, loss_dict, loss_name_suffix='_noise')

        for loss_d in [loss_dict]:
            for ke in loss_d:
                if np.isnan(np.array(loss_d[ke].detach().cpu())):
                    print('!!!!!!!!!  %s is nan' % ke)
                    print(loss_d)
                    raise Exception()

    @staticmethod
    def forward_general_gen(dis_res, loss_dict, loss_name_suffix=''):
        if cfg.gan_type == 'lsgan':
            loss_dis = torch.mean((dis_res - 1) ** 2)
        elif cfg.gan_type == 'nsgan':
            all1 = torch.ones_like(dis_res.data).cuda()
            loss_dis = torch.mean(F.binary_cross_entropy(torch.sigmoid(dis_res), all1))
        elif cfg.gan_type == 'wgan_gp':
            loss_dis = - torch.mean(dis_res)
        elif cfg.gan_type == 'hinge':
            loss_dis = -torch.mean(dis_res)
        elif cfg.gan_type == 'hinge2':
            loss_dis = torch.mean(torch.max(1 - dis_res, torch.zeros_like(dis_res)))
        else:
            raise NotImplementedError()
        loss_dict['lambda_adv' + loss_name_suffix] = loss_dis

    @staticmethod
    def forward_general_dis(dis1, dis0, loss_dict,
                            dis_model=None, input_real=None, input_fake=None, loss_name_suffix=''):
        if cfg.gan_type == 'lsgan':
            loss_dis = torch.mean((dis0 - 0) ** 2) + torch.mean((dis1 - 1) ** 2)
        elif cfg.gan_type == 'nsgan':
            all0 = torch.zeros_like(dis0.data).cuda()
            all1 = torch.ones_like(dis1.data).cuda()
            loss_dis = torch.mean(F.binary_cross_entropy(torch.sigmoid(dis0), all0) +
                                  F.binary_cross_entropy(torch.sigmoid(dis1), all1))
        elif cfg.gan_type == 'wgan_gp':
            loss_dis = torch.mean(dis0) - torch.mean(dis1)
        elif cfg.gan_type == 'hinge' or cfg.gan_type == 'hinge2':
            loss_dis = torch.mean(torch.max(1 - dis1, torch.zeros_like(dis1)))
            loss_dis += torch.mean(torch.max(1 + dis0, torch.zeros_like(dis0)))
        else:
            assert 0, "Unsupported GAN type: {}".format(cfg.gan_type)
        loss_dict['lambda_adv' + loss_name_suffix] = loss_dis

        if cfg.gan_type == 'wgan_gp':
            loss_gp = 0
            alpha_gp = torch.rand(input_real.size(0), *([1] * (len(input_real.shape) - 1))).type_as(input_real)
            x_hat = (alpha_gp * input_real + (1 - alpha_gp) * input_fake).requires_grad_(True)
            out_hat = dis_model.forward(x_hat)
            # gradient penalty
            weight = torch.ones(out_hat.size()).type_as(out_hat)
            dydx = torch.autograd.grad(outputs=out_hat, inputs=x_hat, grad_outputs=weight, retain_graph=True,
                                       create_graph=True, only_inputs=True)[0]
            dydx = dydx.contiguous().view(dydx.size(0), -1)
            dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
            loss_gp += torch.mean((dydx_l2norm - 1) ** 2)
            loss_dict['lambda_gp' + loss_name_suffix] = loss_gp

        if cfg.lambda_gp_0 and cfg.lambda_gp_0 > 0:
            # ii = input_real.requires_grad_(True)
            dydx = torch.autograd.grad(outputs=dis1.sum(), inputs=input_real, retain_graph=True, create_graph=True,
                                       only_inputs=True, allow_unused=True)[0]
            dydx2 = dydx.pow(2)
            dydx_l2norm = dydx2.view(dydx.size(0), -1).sum(1)
            loss_gp = dydx_l2norm.mean()
            loss_dict['lambda_gp_0' + loss_name_suffix] = loss_gp

    def forward_d(self, loss_dict, real_batch):
        if cfg.disturb_real_batch_mask:
            real_batch = self.disturb_real(real_batch)

        if 'lambda_gp_0' in cfg and cfg.lambda_gp_0 > 0:
            real_batch = real_batch.requires_grad_()
        dis_out_real = self.dis.forward(real_batch)
        self.forward_general_dis(dis_out_real, self.dis_out_fake,
                                 loss_dict, self.dis, input_real=real_batch,
                                 input_fake=self.ae_out_mask)

    def disturb_real(self, real_batch):
        cur = (torch.rand(real_batch.shape).to(real_batch.device) * 0.03 + real_batch)
        cur = cur / cur.sum(dim=(1), keepdim=True)
        return cur

    def forward_adv_noise(self, loss_dict):

        input_real = self.real_noise
        if 'lambda_gp_0' in cfg and cfg.lambda_gp_0 > 0:
            input_real = input_real.requires_grad_()

        input_fake = self.ae_mid_hair_code.detach()

        dis1 = self.dis_noise(input_real)
        dis0 = self.dis_noise(input_fake)

        self.forward_general_dis(dis1, dis0, loss_dict, self.dis_noise, input_real=input_real,
                                 input_fake=input_fake, loss_name_suffix='_noise')


def get_hair_face_code(mask_generator, mask):
    mask_batch = mask[None, None, :, :]
    mask_batch = T.functional.resize(mask_batch.long(), (256, 256),
                                     interpolation=T.InterpolationMode.NEAREST)
    mask_one_hot = mask_label_to_one_hot(mask_batch)
    hair, face = split_hair_face(mask_one_hot)
    hair_code = mask_generator.forward_hair_encoder(hair, testing=True)
    face_code = mask_generator.forward_face_encoder(face)
    return face_code, hair_code


def get_new_shape(mask_generator, face_code, new_hair_code):
    out_mask = mask_generator.forward_decode_by_code(new_hair_code, face_code)
    out_mask = mask_one_hot_to_label(out_mask)[0]
    return out_mask
