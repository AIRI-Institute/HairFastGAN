# -*- coding: utf-8 -*-

"""
# File name:    predictor_config.py
# Time :        2021/12/14 21:03
# Author:       xyguoo@163.com
"""

import addict  # nesting dict
import os
import argparse
import sys

from global_value_utils import GLOBAL_DATA_ROOT

configs = [
    addict.Dict({
        "experiment_name": "p002___curliness",
        'basic_dir': 'model_trained/curliness_classifier',
        'filter_female_and_frontal': True,
        'hidden_layer_num': 3,
        'hidden_dim': 32,
        'lambda_cls_curliness': {0: 1, 200: 0.1, 400: 0.01, 2500: 0.001},
        'init_type': 'xavier',
        'norm': 'bn',
        'dropout': 0.5,
        'total_batch_size': 256,
        'total_step': 7000,
    }),
    addict.Dict({
        "experiment_name": "p004___pca_std",
        'basic_dir': 'model_trained/color_encoder',
        'filter_female_and_frontal': True,
        'hidden_layer_num': 3,
        'hidden_dim': 256,
        'lambda_rgb': {0: 1, 7000: 1},
        'lambda_pca_std': {0: 1, 7000: 1},
        'init_type': 'xavier',
        'norm': 'bn',
        'dropout': 0.2,
        'total_batch_size': 256,
        'total_step': 10000,
    }),
]


def get_config(configs, config_id):
    for c in configs:
        if c.experiment_name.startswith(config_id):
            check_add_default_value_to_base_cfg(c)
            return c


def check_add_default_value_to_base_cfg(cfg):
    add_default_value_to_cfg(cfg, 'lr', 0.002)
    add_default_value_to_cfg(cfg, 'beta1', 0.5)
    add_default_value_to_cfg(cfg, 'beta2', 0.999)

    add_default_value_to_cfg(cfg, 'log_step', 10)
    add_default_value_to_cfg(cfg, 'model_save_step', 1000)
    add_default_value_to_cfg(cfg, 'sample_batch_size', 100)
    add_default_value_to_cfg(cfg, 'max_save', 2)
    add_default_value_to_cfg(cfg, 'SEAN_code', 512)

    # Model configuration
    add_default_value_to_cfg(cfg, 'total_batch_size', 64)
    add_default_value_to_cfg(cfg, 'gan_type', 'wgan_gp')

    add_default_value_to_cfg(cfg, 'norm', 'none')
    add_default_value_to_cfg(cfg, 'activ', 'lrelu')
    add_default_value_to_cfg(cfg, 'init_type', 'normal')

    add_default_value_to_cfg(cfg, 'root_dir', '%s/%s' % (cfg.basic_dir, cfg['experiment_name']))
    add_default_value_to_cfg(cfg, 'log_dir', cfg.root_dir + '/logs')
    add_default_value_to_cfg(cfg, 'model_save_dir', cfg.root_dir + '/models')
    add_default_value_to_cfg(cfg, 'sample_dir', cfg.root_dir + '/samples')
    try:
        add_default_value_to_cfg(cfg, 'gpu_num', len(args.gpu.split(',')))
    except:
        add_default_value_to_cfg(cfg, 'gpu_num', 1)

    add_default_value_to_cfg(cfg, 'data_root', GLOBAL_DATA_ROOT)


def add_default_value_to_cfg(cfg, key, value):
    if key not in cfg:
        cfg[key] = value


def merge_config_in_place(ori_cfg, new_cfg):
    for k in new_cfg:
        ori_cfg[k] = new_cfg[k]


def back_process(cfg):
    cfg.batch_size = cfg.total_batch_size // cfg.gpu_num
    cfg.predict_dict = {}
    if 'lambda_cls_curliness' in cfg:
        cfg.predict_dict['cls_curliness'] = 1
    if 'lambda_rgb' in cfg:
        cfg.predict_dict['rgb_mean'] = 3
    if 'lambda_pca_std' in cfg:
        cfg.predict_dict['pca_std'] = 1


def get_basic_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Specify config number', default='000')
    parser.add_argument('-g', '--gpu', type=str, help='Specify GPU number', default='0')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser


if sys.argv[0].endswith('predictor_train.py'):
    parser = get_basic_arg_parser()
    args = parser.parse_args()
    cfg = get_config(configs, args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    back_process(cfg)
