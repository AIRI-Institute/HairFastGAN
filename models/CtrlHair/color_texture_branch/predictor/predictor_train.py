# -*- coding: utf-8 -*-

"""
# File name:    predictor_train.py
# Time :        2021/12/14 20:58
# Author:       xyguoo@163.com
# Description:  
"""

import sys

sys.path.append('.')

import tensorboardX
import torch
import tqdm
import numpy as np
from color_texture_branch.predictor.predictor_config import cfg, args
from color_texture_branch.dataset import Dataset
import my_pylib
# distributed training
import torch.distributed as dist
from color_texture_branch.predictor.predictor_solver import PredictorSolver
from my_torchlib.train_utils import LossUpdater, to_device, train
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

    # Loss class
    solver = PredictorSolver(cfg, device, local_rank, training=True)

    loss_updater = LossUpdater(cfg)

    # load checkpoint
    ckpt_dir = out_dir + '/checkpoints'
    if local_rank <= 0:
        my_pylib.mkdir(out_dir)
        my_pylib.save_json(out_dir + '/setting_hair.json', cfg, indent=4, separators=(',', ': '))
        my_pylib.mkdir(ckpt_dir)

    try:
        ckpt = my_torchlib.load_checkpoint(ckpt_dir)
        start_step = ckpt['step'] + 1
        for model_name in ['Predictor']:
            cur_model = ckpt[model_name]
            if list(cur_model)[0].startswith('module'):
                ckpt[model_name] = {kk[7:]: cur_model[kk] for kk in cur_model}
        solver.pred.load_state_dict(ckpt['Predictor'], strict=True)
        solver.optimizer.load_state_dict(ckpt['optimizer'])
        print('Load succeed!')
    except:
        print(' [*] No checkpoint!')
        init_weights(solver.pred, init_type=cfg.init_type)
        start_step = 1

    # writer
    if local_rank <= 0:
        writer = tensorboardX.SummaryWriter(out_dir + '/summaries')
    else:
        writer = None

    # start training
    test_batch = ds.get_testing_batch(100)
    test_batch_curliness = ds.get_curliness_hair_test()

    to_device(test_batch_curliness, device)
    to_device(test_batch, device)

    if local_rank >= 0:
        dist.barrier()

    total_step = cfg.total_step + 2
    for step in tqdm.tqdm(range(start_step, total_step), total=total_step, initial=start_step, desc='step'):
        loss_updater.update(step)
        write_log = (writer and step % 11 == 0)

        loss_dict = {}
        if 'rgb_mean' in cfg.predict_dict or 'pca_std' in cfg.predict_dict:
            data = ds.get_training_batch(cfg.batch_size)
            to_device(data, device)
            solver.forward(data)
            solver.forward_d(loss_dict)
            if write_log:
                solver.pred.eval()
                if 'rgb_mean' in cfg.predict_dict:
                    loss_dict['test_lambda_rgb'] = solver.mse_loss(
                        solver.pred(test_batch)['rgb_mean'], test_batch['rgb_mean'])
                    print('rgb loss: %f' % loss_dict['test_lambda_rgb'])
                if 'pca_std' in cfg.predict_dict:
                    loss_dict['test_lambda_pca_std'] = solver.mse_loss(
                        solver.pred(test_batch)['pca_std'], test_batch['pca_std'])
                    print('pca_std loss: %f' % loss_dict['test_lambda_pca_std'])
                solver.pred.train()

        if cfg.lambda_cls_curliness:
            data = {}
            curliness_label = torch.tensor(np.random.choice([-1, 1], (cfg.batch_size, 1)))
            data['curliness_label'] = curliness_label
            data_curliness = ds.get_curliness_hair(curliness_label)
            to_device(data_curliness, device)
            solver.forward(data_curliness)
            solver.forward_d_curliness(data_curliness, loss_dict)

            # validation to show whether over-fit
            if write_log:
                solver.pred.eval()
                logit = solver.pred(test_batch_curliness)['cls_curliness']
                loss_dict['test_cls_curliness'] = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, test_batch_curliness['curliness_label'] / 2 + 0.5)

                print('cls_curliness: %f' % loss_dict['test_cls_curliness'])
                print('acc %f' % ((logit * test_batch_curliness['curliness_label'] > 0).sum() / logit.shape[0]))
                solver.pred.train()

        train(cfg, loss_dict, optimizers=[solver.optimizer],
              step=step, writer=writer, flag='Pred', write_log=write_log)

        if step > 0 and step % cfg.model_save_step == 0:
            if local_rank <= 0:
                save_model(step, solver, ckpt_dir)
            if local_rank >= 0:
                dist.barrier()


def save_model(step, solver, ckpt_dir):
    save_dic = {'step': step,
                'Predictor': solver.pred.state_dict(),
                'optimizer': solver.optimizer.state_dict()}
    my_torchlib.save_checkpoint(save_dic, '%s/%07d.ckpt' % (ckpt_dir, step), max_keep=cfg.max_save)


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    # mp.spawn(worker, nprocs=cfg.gpu_num, args=(cfg.gpu_num, args))
    worker(proc=None, nprocs=None, args=args)
