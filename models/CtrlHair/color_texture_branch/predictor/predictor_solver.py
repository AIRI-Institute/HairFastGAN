# -*- coding: utf-8 -*-

"""
# File name:    predictor_solver.py
# Time :        2021/12/14 21:57
# Author:       xyguoo@163.com
# Description:  
"""

import torch
import torch.nn.functional as F
from color_texture_branch.predictor.predictor_config import cfg
from color_texture_branch.predictor.predictor_model import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP


class PredictorSolver:

    def __init__(self, cfg, device, local_rank, training=True):
        self.mse_loss = torch.nn.MSELoss()
        self.cfg = cfg

        # model
        self.pred = Predictor(cfg)
        self.pred.to(device)
        self.optimizer = torch.optim.Adam(self.pred.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                          weight_decay=0.00)

        if local_rank >= 0:
            pDDP = lambda m, find_unused: DDP(m, device_ids=[local_rank], output_device=local_rank,
                                              find_unused_parameters=False)
            self.pred = pDDP(self.pred, find_unused=True)
        self.local_rank = local_rank
        self.device = device

    def forward(self, data):
        self.data = data
        self.pred_res = self.pred(data)

    def forward_d(self, loss_dict):
        if 'lambda_rgb' in cfg:
            loss_dict['lambda_rgb'] = self.mse_loss(self.pred_res['rgb_mean'], self.data['rgb_mean'])
        if 'lambda_pca_std' in cfg:
            d_pca_std = self.pred_res['pca_std']
            loss_dict['lambda_pca_std'] = self.mse_loss(d_pca_std, self.data['pca_std'])

    def forward_d_curliness(self, data_curliness, loss_dict):
        if cfg.lambda_cls_curliness:
            cls_curliness = self.pred_res['cls_curliness']
            loss_dict['lambda_cls_curliness'] = F.binary_cross_entropy(
                torch.sigmoid(cls_curliness), data_curliness['curliness_label'].float() / 2 + 0.5)
