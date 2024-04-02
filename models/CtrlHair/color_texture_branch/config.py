# -*- coding: utf-8 -*-

"""
# File name:    config.py
# Time :        2021/11/17 13:10
# Author:       xyguoo@163.com
# Description:  
"""

import addict  # nesting dict
import os
import argparse

from global_value_utils import GLOBAL_DATA_ROOT, DEFAULT_CONFIG_COLOR_TEXTURE_BRANCH

configs = [
    addict.Dict({
        "experiment_name": "045__color_texture_final",
        'lambda_rgb': 0.01,
        'lambda_pca_std': 0.01,
        'noise_dim': 8,
        'filter_female_and_frontal': True,
        'g_hidden_layer_num': 4,
        'd_hidden_layer_num': 4,
        'lambda_moment_1': 0.01,
        'lambda_moment_2': 0.01,
        'lambda_cls_curliness': {0: 0.1},
        'lambda_info_curliness': 1.0,
        'lambda_info': 1.0,
        'curliness_dim': 1,
        'predictor': {'curliness': 'p002', 'rgb': 'p004'},
        'gan_input_from_encoder_prob': 0.3,
        'curliness_with_weight': True,
        'lambda_rec': 1000.0,
        'lambda_rec_img': {0: 0, 600000: 1000},
        'gen_mode': 'eigengan',
        'lambda_orthogonal': 0.1,
    }),
]


def get_config(configs, config_id):
    for c in configs:
        if c.experiment_name.startswith(config_id):
            check_add_default_value_to_base_cfg(c)
            return c
    cfg = addict.Dict({})
    check_add_default_value_to_base_cfg(cfg)
    return cfg


def check_add_default_value_to_base_cfg(cfg):
    add_default_value_to_cfg(cfg, 'lr_d', 0.0002)
    add_default_value_to_cfg(cfg, 'lr_g', 0.0002)
    add_default_value_to_cfg(cfg, 'beta1', 0.5)
    add_default_value_to_cfg(cfg, 'beta2', 0.999)

    add_default_value_to_cfg(cfg, 'total_step', 650100)
    add_default_value_to_cfg(cfg, 'log_step', 10)
    add_default_value_to_cfg(cfg, 'sample_step', 25000)
    add_default_value_to_cfg(cfg, 'model_save_step', 20000)
    add_default_value_to_cfg(cfg, 'sample_batch_size', 32)
    add_default_value_to_cfg(cfg, 'max_save', 2)
    add_default_value_to_cfg(cfg, 'vae_var_output', 'var')
    add_default_value_to_cfg(cfg, 'SEAN_code', 512)

    # Model configuration
    add_default_value_to_cfg(cfg, 'total_batch_size', 128)
    add_default_value_to_cfg(cfg, 'g_hidden_layer_num', 4)
    add_default_value_to_cfg(cfg, 'd_hidden_layer_num', 4)
    add_default_value_to_cfg(cfg, 'd_noise_hidden_layer_num', 3)
    add_default_value_to_cfg(cfg, 'g_hidden_dim', 256)
    add_default_value_to_cfg(cfg, 'd_hidden_dim', 256)
    add_default_value_to_cfg(cfg, 'gan_type', 'wgan_gp')
    add_default_value_to_cfg(cfg, 'lambda_gp', 10.0)
    add_default_value_to_cfg(cfg, 'lambda_adv', 1.0)

    add_default_value_to_cfg(cfg, 'noise_dim', 8)
    add_default_value_to_cfg(cfg, 'g_norm', 'none')
    add_default_value_to_cfg(cfg, 'd_norm', 'none')
    add_default_value_to_cfg(cfg, 'g_activ', 'relu')
    add_default_value_to_cfg(cfg, 'd_activ', 'lrelu')
    add_default_value_to_cfg(cfg, 'init_type', 'normal')
    add_default_value_to_cfg(cfg, 'G_D_train_num', {'G': 1, 'D': 1}, )

    output_root_dir = 'model_trained/color_texture/%s' % cfg['experiment_name']
    add_default_value_to_cfg(cfg, 'root_dir', output_root_dir)
    add_default_value_to_cfg(cfg, 'log_dir', output_root_dir + '/logs')
    add_default_value_to_cfg(cfg, 'model_save_dir', output_root_dir + '/models')
    add_default_value_to_cfg(cfg, 'sample_dir', output_root_dir + '/samples')
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
    if cfg.predictor:
        from color_texture_branch.predictor import predictor_config
        for ke in cfg.predictor:
            pred_cfg = predictor_config.get_config(predictor_config.configs, cfg.predictor[ke])
            predictor_config.back_process(pred_cfg)
            cfg.predictor[ke] = pred_cfg

    if 'gen_mode' in cfg and cfg.gen_mode is 'eigengan':
        cfg.subspace_dim = cfg.noise_dim // cfg.g_hidden_layer_num


def get_basic_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Specify config number', default=DEFAULT_CONFIG_COLOR_TEXTURE_BRANCH)
    parser.add_argument('-g', '--gpu', type=str, help='Specify GPU number', default='0')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser


import sys

if sys.argv[0].endswith('color_texture_branch/scripts.py'):
    parser = get_basic_arg_parser()
    args = parser.parse_args()
    cfg = get_config(configs, args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    back_process(cfg)
else:
    cfg = get_config(configs, DEFAULT_CONFIG_COLOR_TEXTURE_BRANCH)
    # cfg = get_config(configs, '046')
    back_process(cfg)
