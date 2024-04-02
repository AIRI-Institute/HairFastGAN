# -*- coding: utf-8 -*-

"""
# File name:    predictor_model.py
# Time :        2021/12/14 20:47
# Author:       xyguoo@163.com
# Description:  
"""

import torch.nn as nn
from my_torchlib.module import LinearBlock


class Predictor(nn.Module):
    """Discriminator network."""

    def __init__(self, cfg):
        super(Predictor, self).__init__()
        self.cfg = cfg
        layers = [LinearBlock(cfg.SEAN_code, cfg.hidden_dim, cfg.norm, activation=cfg.activ, dropout=cfg.dropout)]
        for _ in range(cfg.hidden_layer_num - 1):
            layers.append(LinearBlock(cfg.hidden_dim, cfg.hidden_dim, cfg.norm, activation=cfg.activ,
                                      dropout=cfg.dropout))

        output_dim = 0
        for ke in cfg.predict_dict:
            output_dim += cfg.predict_dict[ke]
        layers.append(LinearBlock(cfg.hidden_dim, output_dim, 'none', 'none'))
        self.net = nn.Sequential(*layers)
        self.input_name = 'code'

    def forward(self, data_in):
        x = data_in[self.input_name]
        out = self.net(x)
        ptr = 0
        data = {}
        for ke in self.cfg.predict_dict:
            cur_dim = self.cfg.predict_dict[ke]
            data[ke] = out[:, ptr:(ptr + cur_dim)]
            ptr += cur_dim
        return data
