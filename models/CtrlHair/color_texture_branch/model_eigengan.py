# -*- coding: utf-8 -*-

"""
# File name:    model_eigengan.py
# Time :        2021/12/28 21:57
# Author:       xyguoo@163.com
# Description:  
"""

import torch.nn as nn
import torch


class SubspaceLayer(nn.Module):
    def __init__(self, dim: int, n_basis: int):
        super().__init__()
        self.U = nn.Parameter(torch.empty(n_basis, dim), requires_grad=True)
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]), requires_grad=True)
        self.mu = nn.Parameter(torch.zeros(dim), requires_grad=True)

        self.unit_matrix = torch.eye(n_basis, requires_grad=False)

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu

    def orthogonal_regularizer(self):
        UUT = self.U @ self.U.t()
        self.unit_matrix = self.unit_matrix.to(self.U.device)
        reg = ((UUT - self.unit_matrix) ** 2).mean()
        return reg


class EigenGenerator(nn.Module):
    def __init__(self, cfg):
        super(EigenGenerator, self).__init__()
        self.cfg = cfg

        input_dim = 0
        if cfg.lambda_rgb:
            input_dim += 3
        if cfg.lambda_pca_std:
            input_dim += 1
        if cfg.lambda_cls_curliness:
            input_dim += cfg.curliness_dim

        self.main_layer_in = nn.Linear(input_dim, cfg.g_hidden_dim, bias=True)
        main_layers_mid = []
        for _ in range(cfg.g_hidden_layer_num - 1):
            main_layers_mid.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(cfg.g_hidden_dim,
                                                                              cfg.g_hidden_dim, bias=True)))
        main_layers_mid.append(nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(cfg.g_hidden_dim, cfg.SEAN_code)))

        subspaces = []
        for _ in range(cfg.g_hidden_layer_num):
            sub = SubspaceLayer(cfg.g_hidden_dim, cfg.subspace_dim)
            subspaces.append(sub)

        self.main_layer_mid = nn.ModuleList(main_layers_mid)
        self.subspaces = nn.ModuleList(subspaces)

    def forward(self, data):
        noise = data['noise']
        noise = noise.reshape(len(data['noise']), self.cfg.g_hidden_layer_num, self.cfg.subspace_dim)

        input_data = []
        if self.cfg.lambda_cls_curliness:
            input_data.append(data['noise_curliness'])
        if self.cfg.lambda_rgb:
            input_data.append(data['rgb_mean'])
        if self.cfg.lambda_pca_std:
            input_data.append(data['pca_std'])

        x = torch.cat(input_data, dim=1)
        x_mid = self.main_layer_in(x)

        for layer_idx in range(self.cfg.g_hidden_layer_num):
            subspace_data = self.subspaces[layer_idx](noise[:, layer_idx, :])
            x_mid = x_mid + subspace_data
            x_mid = self.main_layer_mid[layer_idx](x_mid)

        res = {'code': x_mid}
        return res

    def orthogonal_regularizer_loss(self):
        loss = 0
        for s in self.subspaces:
            loss = loss + s.orthogonal_regularizer()
        return loss
