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

from models.CtrlHair.global_value_utils import GLOBAL_DATA_ROOT, DEFAULT_CONFIG_SHAPE_BRANCH

configs = [
    addict.Dict({
        "experiment_name": "054__succeed__049__gan_fake_0.5_from_noise",
        'hair_dim': 16,
        'pos_encoding_order': 10,
        'lambda_hair': 100,
        'lambda_non_hair': 100,
        'lambda_face': 20,
        'lambda_self_rec': 5,
        'lambda_kl': 0.1,
        'regular_method': 'ce',
        'full_dataset': True,
        'only_celeba_as_real': True,
        'g_norm': 'ln',
        'd_norm': 'none',
        'lr_g': 0.0002,
        'lambda_adv_noise': 1,
        'lambda_gp_0_noise': 10,
        'total_batch_size': 4,
        'random_ae_prob': 0.5,
        'lr_dz': 0.00005,
        'adaptor_test_pool_dir': 'shape_testing_wrap_pool',
        'adaptor_pool_dir': 'shape_training_wrap_pool'
    }),
]


def get_config(configs, config_id):
    for c in configs:
        if c.experiment_name.startswith(config_id):
            check_add_default_value_to_base_cfg(c)
            return c


def check_add_default_value_to_base_cfg(cfg):
    add_default_value_to_cfg(cfg, 'lr_d', 0.0001)
    add_default_value_to_cfg(cfg, 'lr_g', 0.0002)
    add_default_value_to_cfg(cfg, 'lr_dz', 0.0001)
    add_default_value_to_cfg(cfg, 'beta1', 0.5)
    add_default_value_to_cfg(cfg, 'beta2', 0.999)

    add_default_value_to_cfg(cfg, 'total_step', 380002)
    add_default_value_to_cfg(cfg, 'log_step', 10)
    add_default_value_to_cfg(cfg, 'sample_step', 10000)
    add_default_value_to_cfg(cfg, 'model_save_step', 10000)
    add_default_value_to_cfg(cfg, 'sample_batch_size', 16)
    add_default_value_to_cfg(cfg, 'max_save', 1)
    add_default_value_to_cfg(cfg, 'vae_var_output', 'var')
    add_default_value_to_cfg(cfg, 'SEAN_code', 512)
    add_default_value_to_cfg(cfg, 'd_hidden_in_channel', 16)

    # Model configuration
    add_default_value_to_cfg(cfg, 'total_batch_size', 4)
    add_default_value_to_cfg(cfg, 'gan_type', 'hinge2')
    add_default_value_to_cfg(cfg, 'lambda_gp_0', 10.0)
    add_default_value_to_cfg(cfg, 'lambda_adv', 1.0)

    add_default_value_to_cfg(cfg, 'g_norm', 'bn')
    add_default_value_to_cfg(cfg, 'd_norm', 'bn')
    add_default_value_to_cfg(cfg, 'init_type', 'normal')
    add_default_value_to_cfg(cfg, 'G_D_train_num', {'G': 1, 'D': 1}, )
    add_default_value_to_cfg(cfg, 'vae_hair_mode', True)

    output_root_dir = 'model_trained/shape/%s' % cfg['experiment_name']
    add_default_value_to_cfg(cfg, 'root_dir', output_root_dir)
    add_default_value_to_cfg(cfg, 'log_dir', output_root_dir + '/summaries')
    add_default_value_to_cfg(cfg, 'checkpoints_dir', output_root_dir + '/checkpoints')
    add_default_value_to_cfg(cfg, 'sample_dir', output_root_dir + '/sample_training')

    try:
        add_default_value_to_cfg(cfg, 'gpu_num', len(args.gpu.split(',')))
    except:
        add_default_value_to_cfg(cfg, 'gpu_num', 1)
    add_default_value_to_cfg(cfg, 'img_size', 256)
    add_default_value_to_cfg(cfg, 'data_root', GLOBAL_DATA_ROOT)

    # dz discriminator
    add_default_value_to_cfg(cfg, 'd_hidden_dim', 256)
    add_default_value_to_cfg(cfg, 'd_noise_hidden_layer_num', 3)


def add_default_value_to_cfg(cfg, key, value):
    if key not in cfg:
        cfg[key] = value


def merge_config_in_place(ori_cfg, new_cfg):
    for k in new_cfg:
        ori_cfg[k] = new_cfg[k]


def back_process(cfg):
    cfg.batch_size = cfg.total_batch_size // cfg.gpu_num


def get_basic_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Specify config number', default=DEFAULT_CONFIG_SHAPE_BRANCH)
    parser.add_argument('-g', '--gpu', type=str, help='Specify GPU number', default='0')
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser


import sys

if sys.argv[0].endswith('shape_branch/scripts.py') or sys.argv[0].endswith('shape_branch/script_find_direction.py'):
    parser = get_basic_arg_parser()
    args = parser.parse_args()
    cfg = get_config(configs, args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    back_process(cfg)
else:
    cfg = get_config(configs, DEFAULT_CONFIG_SHAPE_BRANCH)
    back_process(cfg)
