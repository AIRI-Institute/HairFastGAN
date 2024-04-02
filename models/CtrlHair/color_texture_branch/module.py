# -*- coding: utf-8 -*-

"""
# File name:    module.py
# Time :        2021/11/17 15:38
# Author:       xyguoo@163.com
# Description:  
"""

import torch.nn as nn


class LinearBlock(nn.Module):

    def __init__(self, input_dim, output_dim, norm, activation='relu', use_bias=True, leaky_slope=0.2, dropout=0):
        super(LinearBlock, self).__init__()
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(leaky_slope, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.dropout = dropout
        if bool(self.dropout) and self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        if bool(self.dropout) and self.dropout > 0:
            out = self.dropout_layer(out)
        return out
