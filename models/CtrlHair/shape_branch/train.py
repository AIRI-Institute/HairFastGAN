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
from shape_branch.config import cfg, args
from shape_branch.dataset import Dataset
import my_pylib
from shape_branch.validation_in_train import print_val_save_model
# distributed training
import torch.distributed as dist
from shape_branch.solver import Solver
from my_torchlib.train_utils import LossUpdater, to_device, generate_noise, train
import my_torchlib
from shape_branch.model import init_weights
from shape_branch.shape_util import mask_label_to_one_hot


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
    ckpt_dir = cfg.checkpoints_dir
    if local_rank <= 0:
        my_pylib.mkdir(out_dir)
        my_pylib.save_json(out_dir + '/setting_hair.json', cfg, indent=4, separators=(',', ': '))
        my_pylib.mkdir(ckpt_dir)
        my_pylib.mkdir(cfg.sample_dir)

    try:
        ckpt = my_torchlib.load_checkpoint(ckpt_dir)
        start_step = ckpt['step'] + 1
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
        start_step = 1

    # writer
    if local_rank <= 0:
        writer = tensorboardX.SummaryWriter(cfg.log_dir)
    else:
        writer = None

    # start training
    test_batch = ds.get_test_batch(cfg.sample_batch_size)
    for ke in ['face', 'target', 'hair']:
        test_batch[ke] = mask_label_to_one_hot(test_batch[ke])
    to_device(test_batch, device)

    if local_rank >= 0:
        dist.barrier()

    total_step = cfg.total_step + 2
    for step in tqdm.tqdm(range(start_step, total_step), total=total_step, initial=start_step, desc='step'):
        loss_updater.update(step)
        write_log = (writer and step % 23 == 0)

        for i in range(sum(cfg.G_D_train_num.values())):
            data = ds.get_random_pair_batch(cfg.batch_size)
            for ke in data:
                data[ke] = mask_label_to_one_hot(data[ke])
            to_device(data, device)
            loss_dict = {}
            solver.forward(data)
            if i < cfg.G_D_train_num['D']:
                real_batch = ds.get_random_single_batch(cfg.batch_size)
                real_batch = mask_label_to_one_hot(real_batch)
                real_batch = real_batch.to(device)
                solver.forward_d(loss_dict, real_batch)
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

        print_val_save_model(step, cfg.sample_dir, solver, test_batch, ckpt_dir, local_rank)


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    # mp.spawn(worker, nprocs=cfg.gpu_num, args=(cfg.gpu_num, args))
    worker(proc=None, nprocs=None, args=args)
