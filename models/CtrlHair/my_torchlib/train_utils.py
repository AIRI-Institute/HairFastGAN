# -*- coding: utf-8 -*-

"""
# File name:    train_utils.py
# Time :        2021/12/15 11:21
# Author:       xyguoo@163.com
# Description:  
"""
import numpy as np
import torch
from scipy import stats

class LossUpdater:
    def __init__(self, cfg):
        self.cfg = cfg
        self.register_list = {}
        for ke in cfg:
            if ke.startswith('lambda_') and isinstance(cfg[ke], dict):
                self.register_list[ke] = cfg[ke]

    def update(self, step):
        for ke in self.register_list:
            loss_dict = self.register_list[ke]
            weight = None
            for start_step in loss_dict:
                if start_step > step:
                    break
                weight = loss_dict[start_step]
            if weight is None:
                raise Exception()
            self.cfg[ke] = weight


def tensor2numpy(tensor):
    return tensor.cpu().numpy().transpose(1, 2, 0)


def to_device(dat, device):
    for ke in dat:
        if isinstance(dat[ke], torch.Tensor):
            dat[ke] = dat[ke].to(device)


def generate_noise(bs, dim, label=None):
    # trunc = stats.truncnorm(-3, 3)
    # noise = trunc.rvs(bs * dim).reshape(bs, dim)
    # noise = torch.tensor(noise).float()
    noise = torch.randn((bs, dim))
    if label is not None:
        noise = (noise.abs() * label).float()
    return noise


def train(cfg, loss_dict, optimizers, step, writer, flag, retain_graph=False, write_log=False):
    """
    :param loss_dict:
    :param optimizers:
    :param step:
    :param writer:
    :param flag:
    :return:
    """
    if len(loss_dict) == 0:
        return
    loss_total = 0
    for k, v in loss_dict.items():
        if np.isnan(np.array(v.detach().cpu())):
            print('!!!!!!!!!  %s is nan' % k)
            raise Exception()
        if np.isinf(np.array(v.detach().cpu())):
            print('!!!!!!!!!  %s is inf' % k)
            raise Exception()
        if k not in cfg:  # skip rgs_zp
            continue
        loss_total = loss_total + v * cfg[k]

    for o in optimizers:
        o.zero_grad()
    loss_total.backward(retain_graph=retain_graph)
    for o in optimizers:
        o.step()

    # summary
    if write_log:
        for k, v in loss_dict.items():
            writer.add_scalar('%s/%s' % (flag, k), loss_dict[k].data.mean().cpu().numpy(),
                              global_step=step)
        writer.add_scalar('%s/%s' % (flag, 'total'), loss_total.data.mean().cpu().numpy(),
                          global_step=step)