# -*- coding: utf-8 -*-

"""
# File name:    solver.py
# Time :        2021/11/17 16:24
# Author:       xyguoo@163.com
# Description:  
"""
import torch
import torch.nn.functional as F
import numpy as np
from .config import cfg
from .model import Discriminator, DiscriminatorNoise
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import os
import cv2


# solver
class Solver:

    def __init__(self, cfg, device, local_rank, training=True):
        self.mse_loss = torch.nn.MSELoss()
        self.cfg = cfg

        # model
        if 'gen_mode' in cfg and cfg.gen_mode is 'eigengan':
            from color_texture_branch.model_eigengan import EigenGenerator
            self.gen = EigenGenerator(cfg)
        else:
            from color_texture_branch.model import Generator
            self.gen = Generator(cfg)
        self.gen.to(device)

        self.dis = Discriminator(cfg)
        self.dis.to(device)

        if 'curliness' in cfg.predictor:
            from color_texture_branch.predictor.predictor_model import Predictor
            self.curliness_model = Predictor(cfg.predictor['curliness'])
            self.curliness_model.to(device)
            self.curliness_model.eval()

        if 'rgb' in cfg.predictor:
            from color_texture_branch.predictor.predictor_model import Predictor
            self.rgb_model = Predictor(cfg.predictor['rgb'])
            self.rgb_model.to(device)
            self.rgb_model.eval()

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
            if cfg.lambda_adv_noise:
                self.dis_noise = pDDP(self.dis_noise, find_unused=True)
        self.local_rank = local_rank
        self.device = device

    def edit_infer(self, hair_code, data):
        self.inner_code = self.dis({'code': hair_code})
        for ke in data:
            self.inner_code[ke] = data[ke]
        self.res = self.gen(self.inner_code)
        return self.res['code']

    def forward(self, data):
        self.ae_in = {'code': data['code']}

        # rec
        d_res_real = self.dis(self.ae_in)
        self.ae_mid = {'noise': d_res_real['noise'], 'rgb_mean': data['rgb_mean'], 'pca_std': data['pca_std']}
        if cfg.lambda_cls_curliness:
            self.ae_mid['noise_curliness'] = d_res_real['noise_curliness']
        self.ae_out = self.gen(self.ae_mid)

        # gan
        random_list = list(range(data['rgb_mean'].shape[0]))
        random.shuffle(random_list)
        self.gan_in = {'rgb_mean': data['rgb_mean'][random_list]}
        self.gan_in['pca_std'] = data['pca_std'][random_list]
        random.shuffle(random_list)
        if cfg.lambda_cls_curliness:
            self.gan_in['noise_curliness'] = data['noise_curliness'][random_list]
        self.gan_in['curliness_label'] = data['curliness_label'][random_list]
        random.shuffle(random_list)

        if self.cfg.gan_input_from_encoder_prob and \
                random.random() < self.cfg.gan_input_from_encoder_prob:
            self.gan_in['noise'] = d_res_real['noise'][random_list].detach()
        else:
            self.gan_in['noise'] = data['noise'][random_list]
        self.gan_mid = self.gen(self.gan_in)
        self.gan_out_fake = self.dis(self.gan_mid)
        self.gan_out_real = d_res_real
        self.gan_label = {'rgb_mean': data['rgb_mean'], 'pca_std': data['pca_std'], 'curliness_label': data['curliness_label']}
        self.real_noise = {'noise': data['noise']}
        if cfg.lambda_cls_curliness:
            self.real_noise['noise_curliness'] = data['noise_curliness']

    def forward_g(self, loss_dict):
        self.forward_general_gen(self.gan_out_fake['adv'], loss_dict)

        loss_dict['lambda_info'] = self.mse_loss(self.gan_out_fake['noise'], self.gan_in['noise'])
        loss_dict['lambda_rec'] = self.mse_loss(self.ae_out['code'], self.ae_in['code'])

        if 'rgb' in cfg.predictor:
            p_rgb = self.rgb_model(self.gan_mid)
        if cfg.lambda_rgb:
            if 'rgb' in cfg.predictor:
                d_rgb_mean = p_rgb['rgb_mean']
            else:
                d_rgb_mean = self.gan_out_fake['rgb_mean']
            loss_dict['lambda_rgb'] = self.mse_loss(d_rgb_mean, self.gan_in['rgb_mean'])

        if cfg.lambda_pca_std:
            if 'rgb' in cfg.predictor:
                d_pca_std = p_rgb['pca_std']
            else:
                d_pca_std = self.gan_out_fake['pca_std']
            loss_dict['lambda_pca_std'] = self.mse_loss(d_pca_std, self.gan_in['pca_std'])

        if cfg.lambda_cls_curliness:
            d_noise_curliness = self.gan_out_fake['noise_curliness']
            loss_dict['lambda_info_curliness'] = self.mse_loss(d_noise_curliness, self.gan_in['noise_curliness'])
            if 'curliness' in cfg.predictor:
                cls_curliness = self.curliness_model(self.gan_mid)['cls_curliness']
            else:
                cls_curliness = self.gan_out_fake['cls_curliness']
            if cfg.curliness_with_weight:
                weights = self.gan_in['noise_curliness'].abs()
                weights = weights / weights.sum() * weights.shape[0]
                loss_dict['lambda_cls_curliness'] = F.binary_cross_entropy(torch.sigmoid(cls_curliness),
                                                                      self.gan_in['curliness_label'].float() / 2 + 0.5,
                                                                      weight=weights)
            else:
                loss_dict['lambda_cls_curliness'] = F.binary_cross_entropy(torch.sigmoid(cls_curliness),
                                                                      self.gan_in['curliness_label'].float() / 2 + 0.5)

        if 'gen_mode' in cfg and cfg.gen_mode is 'eigengan':
            loss_dict['lambda_orthogonal'] = self.gen.orthogonal_regularizer_loss()

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
    def forward_general_dis(dis1, dis0, dis_model, loss_dict, input_real, input_fake, loss_name_suffix=''):

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
            loss_dis += torch.mean(torch.max(1 + dis0, torch.zeros_like(dis1)))
        else:
            assert 0, "Unsupported GAN type: {}".format(dis_model.gan_type)
        loss_dict['lambda_adv' + loss_name_suffix] = loss_dis

        if cfg.gan_type == 'wgan_gp':
            loss_gp = 0
            alpha_gp = torch.rand(input_real.size(0), 1, ).type_as(input_real)
            x_hat = (alpha_gp * input_real + (1 - alpha_gp) * input_fake).requires_grad_(True)
            out_hat = dis_model.forward_adv_direct(x_hat)
            # gradient penalty
            weight = torch.ones(out_hat.size()).type_as(out_hat)
            dydx = torch.autograd.grad(outputs=out_hat, inputs=x_hat, grad_outputs=weight, retain_graph=True,
                                       create_graph=True, only_inputs=True)[0]
            dydx = dydx.contiguous().view(dydx.size(0), -1)
            dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
            loss_gp += torch.mean((dydx_l2norm - 1) ** 2)
            loss_dict['lambda_gp' + loss_name_suffix] = loss_gp

    def forward_d(self, loss_dict):
        self.forward_general_dis(self.gan_out_real['adv'], self.gan_out_fake['adv'],
                                 self.dis, loss_dict, input_real=self.ae_in['code'],
                                 input_fake=self.gan_mid['code'])

        loss_dict['lambda_info'] = self.mse_loss(self.gan_out_fake['noise'], self.gan_in['noise'])
        if cfg.lambda_rgb and 'rgb' not in cfg.predictor:
            loss_dict['lambda_rgb'] = self.mse_loss(self.gan_in['rgb_mean'], self.gan_out_fake['rgb_mean'])
        loss_dict['lambda_rec'] = self.mse_loss(self.ae_out['code'], self.ae_in['code'])
        if cfg.lambda_pca_std and 'rgb' not in cfg.predictor:
            loss_dict['lambda_pca_std'] = self.mse_loss(self.gan_out_real['pca_std'], self.gan_label['pca_std'])

        if cfg.lambda_adv_noise:
            self.d_noise_res = self.dis_noise(self.ae_mid)
            self.forward_general_gen(self.d_noise_res['adv'], loss_dict, loss_name_suffix='_noise')

        if cfg.lambda_moment_1 or cfg.lambda_moment_2:
            if cfg.lambda_cls_curliness:
                noise_mid = torch.cat([self.ae_mid['noise_curliness'], self.ae_mid['noise']], dim=1)
            else:
                noise_mid = self.ae_mid['noise']
            if cfg.lambda_moment_1:
                loss_dict['lambda_moment_1'] = (noise_mid.mean(dim=0) ** 2).mean()
            if cfg.lambda_moment_2:
                loss_dict['lambda_moment_2'] = (((noise_mid ** 2).mean(dim=0) - 1) ** 2).mean()

        if cfg.lambda_cls_curliness:
            loss_dict['lambda_info_curliness'] = self.mse_loss(self.gan_out_fake['noise_curliness'], self.gan_in['noise_curliness'])

    def forward_d_curliness(self, data_curliness, loss_dict):
        d_res = self.dis(data_curliness)
        cls_curliness = d_res['cls_curliness']
        loss_dict['lambda_cls_curliness'] = F.binary_cross_entropy(torch.sigmoid(cls_curliness),
                                                              data_curliness['curliness_label'].float() / 2 + 0.5)

    def forward_adv_noise(self, loss_dict):
        dis1 = self.dis_noise(self.real_noise)['adv']
        self.ae_mid['noise'] = self.ae_mid['noise'].detach()
        if self.cfg.lambda_cls_curliness:
            self.ae_mid['noise_curliness'] = self.ae_mid['noise_curliness'].detach()
        self.d_noise_res = self.dis_noise(self.ae_mid)
        dis0 = self.d_noise_res['adv']

        input_real = self.real_noise['noise']
        input_fake = self.ae_mid['noise']
        if self.cfg.lambda_cls_curliness:
            input_real = torch.cat([input_real, self.real_noise['noise_curliness']], dim=1)
            input_fake = torch.cat([input_fake, self.ae_mid['noise_curliness']], dim=1)

        self.forward_general_dis(dis1, dis0, self.dis_noise, loss_dict, input_real=input_real,
                                 input_fake=input_fake, loss_name_suffix='_noise')

    def forward_rec_img(self, data, loss_dict, batch_size=4):
        from .validation_in_train import he
        from global_value_utils import HAIR_IDX

        items = data['items']

        rec_loss = 0
        for idx in range(batch_size):
            item = items[idx]
            dataset_name, img_name = item.split('___')
            parsing_img = cv2.imread(os.path.join(self.cfg.data_root, '%s/label/%s.png') %
                                     (dataset_name, img_name), cv2.IMREAD_GRAYSCALE)
            parsing_img = cv2.resize(parsing_img, (256, 256), cv2.INTER_NEAREST)

            sean_code = torch.tensor(data['sean_code'][idx].copy(), dtype=torch.float32).to(self.ae_out['code'].device)
            sean_code[HAIR_IDX] = self.ae_out['code'][idx, ...]

            render_img = he.gen_img(sean_code[None, ...], parsing_img[None, None, ...])

            input_img = cv2.cvtColor(cv2.imread(os.path.join(self.cfg.data_root, '%s/images_256/%s.png') %
                                                (dataset_name, img_name)), cv2.COLOR_BGR2RGB)
            input_img = input_img / 127.5 - 1.0
            input_img = input_img.transpose(2, 0, 1)

            input_img = torch.tensor(input_img).to(render_img.device)
            parsing_img = torch.tensor(parsing_img).to(render_img.device)

            rec_loss = rec_loss + ((input_img - render_img)[:, (parsing_img == HAIR_IDX)] ** 2).mean()
        rec_loss = rec_loss / batch_size
        loss_dict['lambda_rec_img'] = rec_loss
