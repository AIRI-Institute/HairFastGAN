# -*- coding: utf-8 -*-

"""
# File name:    scripts.py.py
# Time :        2021/11/17 15:24
# Author:       xyguoo@163.com
# Description:  
"""

import sys

sys.path.append('.')

import tensorboardX
import torch
import tqdm
import numpy as np
from color_texture_branch.config import cfg, args
from color_texture_branch.dataset import Dataset
import my_pylib
from color_texture_branch.validation_in_train import print_val_save_model
# distributed training
import torch.distributed as dist
from color_texture_branch.solver import Solver
from my_torchlib.train_utils import LossUpdater, to_device, generate_noise, train
import my_torchlib
from color_texture_branch.model import init_weights


def get_total_step():
    total = 0
    for key in cfg.iter:
        total += cfg.iter[key]
    return total


def worker(proc, nprocs, args):
    local_rank = args.local_rank
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='tcp://localhost:%d' % (6030 + int(cfg.experiment_name[:3])),
                                rank=args.local_rank,
                                world_size=cfg.gpu_num)
        print('setup rank %d' % local_rank)
    device = torch.device('cuda', max(0, local_rank))

    # config
    out_dir = cfg.root_dir

    # data
    ds = Dataset(cfg)

    loss_updater = LossUpdater(cfg)
    loss_updater.update(0)

    # Loss class
    solver = Solver(cfg, device, local_rank=local_rank)

    # load checkpoint
    ckpt_dir = out_dir + '/checkpoints'
    if local_rank <= 0:
        my_pylib.mkdir(out_dir)
        my_pylib.save_json(out_dir + '/setting_hair.json', cfg, indent=4, separators=(',', ': '))
        my_pylib.mkdir(ckpt_dir)

    try:
        ckpt = my_torchlib.load_checkpoint(ckpt_dir)
        start_step = ckpt['step'] + 1
        for model_name in ['Model_G', 'Model_D']:
            cur_model = ckpt[model_name]
            if list(cur_model)[0].startswith('module'):
                ckpt[model_name] = {kk[7:]: cur_model[kk] for kk in cur_model}
        solver.gen.load_state_dict(ckpt['Model_G'], strict=True)
        solver.dis.load_state_dict(ckpt['Model_D'], strict=True)
        solver.D_optimizer.load_state_dict(ckpt['D_optimizer'])
        solver.G_optimizer.load_state_dict(ckpt['G_optimizer'])
        if cfg.lambda_adv_noise:
            solver.dis_noise.load_state_dict(ckpt['Model_D_noise'], strict=True)
            solver.D_noise_optimizer.load_state_dict(ckpt['D_noise_optimizer'])
        print('Load succeed!')
    except:
        print(' [*] No checkpoint!')
        init_weights(solver.gen, init_type=cfg.init_type)
        init_weights(solver.dis, init_type=cfg.init_type)
        if cfg.lambda_adv_noise:
            init_weights(solver.dis_noise, init_type=cfg.init_type)
        start_step = 1

    if 'curliness' in cfg.predictor:
        ckpt = my_torchlib.load_checkpoint(cfg.predictor.curliness.root_dir + '/checkpoints')
        solver.curliness_model.load_state_dict(ckpt['Predictor'], strict=True)

    if 'rgb' in cfg.predictor:
        ckpt = my_torchlib.load_checkpoint(cfg.predictor.rgb.root_dir + '/checkpoints')
        solver.rgb_model.load_state_dict(ckpt['Predictor'], strict=True)

    # writer
    if local_rank <= 0:
        writer = tensorboardX.SummaryWriter(out_dir + '/summaries')
    else:
        writer = None

    # start training
    test_batch = ds.get_testing_batch(cfg.sample_batch_size)
    test_batch_curliness = ds.get_curliness_hair_test()

    to_device(test_batch_curliness, device)
    to_device(test_batch, device)

    if local_rank >= 0:
        dist.barrier()

    total_step = cfg.total_step + 2
    for step in tqdm.tqdm(range(start_step, total_step), total=total_step, initial=start_step, desc='step'):
        loss_updater.update(step)
        write_log = (writer and step % 23 == 0)

        for i in range(sum(cfg.G_D_train_num.values())):
            data = ds.get_training_batch(cfg.batch_size)
            data['noise'] = generate_noise(cfg.batch_size, cfg.noise_dim)
            if cfg.lambda_cls_curliness:
                curliness_label = torch.tensor(np.random.choice([-1, 1], (cfg.batch_size, 1)))
                data['curliness_label'] = curliness_label
                data['noise_curliness'] = generate_noise(cfg.batch_size, cfg.curliness_dim, curliness_label)
            to_device(data, device)
            loss_dict = {}
            solver.forward(data)

            if 'lambda_rec_img' in cfg and cfg.lambda_rec_img > 0:
                solver.forward_rec_img(data, loss_dict)

            if i < cfg.G_D_train_num['D']:
                solver.forward_d(loss_dict)
                if cfg.lambda_cls_curliness and not 'curliness' in cfg.predictor:
                    data_curliness = ds.get_curliness_hair(curliness_label)
                    to_device(data_curliness, device)
                    solver.forward_d_curliness(data_curliness, loss_dict)

                    # validation to show whether over-fit
                    if write_log:
                        loss_dict['test_cls_curliness'] = torch.nn.functional.binary_cross_entropy_with_logits(
                            solver.dis(test_batch_curliness)['cls_curliness'], test_batch_curliness['curliness_label'] / 2 + 0.5)
                if cfg.lambda_rgb and 'rgb' not in cfg.predictor and write_log:
                    loss_dict['test_lambda_rgb'] = solver.mse_loss(solver.dis(test_batch)['rgb_mean'],
                                                                   test_batch['rgb_mean'])
                train(cfg, loss_dict, optimizers=[solver.D_optimizer],
                      step=step, writer=writer, flag='D', write_log=write_log)
            else:
                solver.forward_g(loss_dict)
                train(cfg, loss_dict, optimizers=[solver.G_optimizer],
                      step=step, writer=writer, flag='G', write_log=write_log)

                if cfg.lambda_adv_noise:
                    loss_dict = {}
                    solver.forward_adv_noise(loss_dict)
                    train(cfg, loss_dict, optimizers=[solver.D_noise_optimizer], step=step, writer=writer, flag='D_noise',
                          write_log=write_log)

        print_val_save_model(step, out_dir, solver, test_batch, ckpt_dir, local_rank)


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    # mp.spawn(worker, nprocs=cfg.gpu_num, args=(cfg.gpu_num, args))
    worker(proc=None, nprocs=None, args=args)
