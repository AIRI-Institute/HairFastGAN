# -*- coding: utf-8 -*-

"""
# File name:    validation_in_train.py
# Time :        2021/12/10 12:55
# Author:       xyguoo@163.com
# Description:  
"""

import cv2
import os

from .config import cfg
import my_pylib
import torch
from my_torchlib.train_utils import tensor2numpy, to_device, generate_noise
from util.canvas_grid import Canvas
from global_value_utils import HAIR_IDX
import copy
import numpy as np
import torch.distributed as dist
import my_torchlib
from hair_editor import HairEditor

he = HairEditor(load_feature_model=False, load_mask_model=False)


def gen_by_sean(sean_code, item):
    dataset_name, img_name = item.split('___')
    parsing_img = cv2.imread(os.path.join(cfg.data_root, '%s/label/%s.png') %
                             (dataset_name, img_name), cv2.IMREAD_GRAYSCALE)
    parsing_img = cv2.resize(parsing_img, (256, 256), cv2.INTER_NEAREST)
    return he.gen_img(sean_code[None, ...], parsing_img[None, None, ...])


def save_model(step, solver, ckpt_dir):
    save_dic = {'step': step,
                'Model_G': solver.gen.state_dict(), 'Model_D': solver.dis.state_dict(),
                'D_optimizer': solver.D_optimizer.state_dict(), 'G_optimizer': solver.G_optimizer.state_dict()}
    if cfg.lambda_adv_noise:
        save_dic['Model_D_noise'] = solver.dis_noise.state_dict()
        save_dic['D_noise_optimizer'] = solver.D_noise_optimizer.state_dict()
    my_torchlib.save_checkpoint(save_dic, '%s/%07d.ckpt' % (ckpt_dir, step), max_keep=cfg.max_save)


def print_val_save_model(step, out_dir, solver, test_batch, ckpt_dir, local_rank):
    """
    :param step:
    :param validation_data:
    :param img_size:
    :param alpha:
    :return:
    """
    if step > 0 and step % cfg.sample_step == 0:
        gen = solver.gen
        dis = solver.dis
        local_rank = solver.local_rank
        device = solver.device

        save_dir = out_dir + '/sample_training'
        my_pylib.mkdir(save_dir)
        gen.eval()
        dis.eval()
        # gen.cpu()
        show_batch_size = 10
        row_idxs = list(range(show_batch_size))
        instance_idxs = list(range(show_batch_size))

        with torch.no_grad():
            items = test_batch['items']
            decoder_res = dis({'code': test_batch['code'].cuda()})
            hair_noise = decoder_res['noise'].cpu()
            test_data = {'noise': hair_noise, 'rgb_mean': test_batch['rgb_mean'],
                         'pca_std': test_batch['pca_std']}
            if cfg.lambda_cls_curliness:
                test_data['noise_curliness'] = decoder_res['noise_curliness'].cpu()
            to_device(test_data, device)
            rec_code = gen(test_data)['code']

            # ----------------
            # generate each noise dim
            # ----------------
            # grid_count = 10
            # lin_space = np.linspace(-3, 3, grid_count)
            grid_count = 6
            lin_space = np.linspace(-2.5, 2.5, grid_count)
            for dim_idx in range(cfg.noise_dim):
                canvas = Canvas(len(row_idxs), grid_count + 1)
                for draw_idx, idx in enumerate(row_idxs):
                    item = items[idx]
                    dataset_name, img_name = item.split('___')
                    # generate origin and reconstruction
                    ori_img = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                      (dataset_name, img_name)), cv2.COLOR_BGR2RGB)

                    canvas.process_draw_image(ori_img, draw_idx, 0)

                for grid_idx in range(grid_count):
                    temp_noise = hair_noise.clone()
                    temp_noise[:, dim_idx] = lin_space[grid_idx]
                    data = copy.deepcopy(test_data)
                    data['noise'] = temp_noise
                    to_device(data, device)
                    code = gen(data)['code']

                    for draw_idx, idx in enumerate(row_idxs):
                        cur_code = test_batch['sean_code'][idx].copy()
                        cur_code[HAIR_IDX] = code[idx].cpu().numpy()
                        item = items[idx]
                        out_img = gen_by_sean(cur_code, item)
                        out_img = tensor2numpy(out_img)
                        canvas.process_draw_image(out_img, draw_idx, grid_idx + 1)
                if local_rank <= 0:
                    canvas.write_(os.path.join(save_dir, '%06d_noise_%02d.png' % (step, dim_idx)))

            # -------------
            # direct transfer all content
            # -------------
            canvas = Canvas(len(row_idxs) + 1, len(instance_idxs) + 2)
            for draw_idx, instance_idx in enumerate(instance_idxs):
                item = items[instance_idx]
                dataset_name, img_name = item.split('___')
                # generate origin and reconstruction
                ori_img = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                  (dataset_name, img_name)), cv2.COLOR_BGR2RGB)
                canvas.process_draw_image(ori_img, 0, draw_idx + 2)

            for draw_idx, idx in enumerate(row_idxs):
                item_row = items[idx]
                dataset_name, img_name = item_row.split('___')
                img_row = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                  (dataset_name, img_name)), cv2.COLOR_BGR2RGB)
                canvas.process_draw_image(img_row, draw_idx + 1, 0)
                sean_code = test_batch['sean_code'][idx]
                rec_img = tensor2numpy(gen_by_sean(sean_code, item_row))
                canvas.process_draw_image(rec_img, draw_idx + 1, 1)
                for draw_idx2, instance_idx in enumerate(instance_idxs):
                    cur_code = test_batch['sean_code'][idx].copy()
                    cur_code[HAIR_IDX] = test_batch['sean_code'][instance_idx][HAIR_IDX]
                    res_img = gen_by_sean(cur_code, item_row)
                    res_img = tensor2numpy(res_img)
                    canvas.process_draw_image(res_img, draw_idx + 1, draw_idx2 + 2)
            if local_rank <= 0:
                canvas.write_(os.path.join(save_dir, 'rgb_direct.png'))

            # -----------
            # random choice
            # -----------
            grid_count = 10
            # generate each noise dim
            canvas = Canvas(len(row_idxs), grid_count + 2)
            for draw_idx, idx in enumerate(row_idxs):
                item = items[idx]
                dataset_name, img_name = item.split('___')
                # generate origin and reconstruction
                ori_img = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                  (dataset_name, img_name)), cv2.COLOR_BGR2RGB)

                canvas.process_draw_image(ori_img, draw_idx, 0)
                cur_code = test_batch['sean_code'][idx].copy()
                cur_code[HAIR_IDX] = rec_code[idx].cpu().numpy()
                rec_img = tensor2numpy(gen_by_sean(cur_code, item))
                canvas.process_draw_image(rec_img, draw_idx, 1)

            temp_noise = generate_noise(grid_count, cfg.noise_dim)
            for grid_idx in range(grid_count):
                data = copy.deepcopy(test_data)
                data['noise'] = torch.tile(temp_noise[[grid_idx]], [test_batch['rgb_mean'].shape[0], 1])
                to_device(data, device)
                code = gen(data)['code']

                for draw_idx, idx in enumerate(row_idxs):
                    cur_code = test_batch['sean_code'][idx].copy()
                    cur_code[HAIR_IDX] = code[idx].cpu().numpy()
                    item = items[idx]
                    out_img = gen_by_sean(cur_code, item)
                    out_img = tensor2numpy(out_img)
                    canvas.process_draw_image(out_img, draw_idx, grid_idx + 2)

            if local_rank <= 0:
                canvas.write_(os.path.join(save_dir, '%06d_random.png' % step))

            # ------------
            # generate curliness
            # ------------
            if cfg.lambda_cls_curliness:
                grid_count = 10
                lin_space = np.linspace(-3, 3, grid_count)
                canvas = Canvas(len(row_idxs), grid_count + 2)
                for draw_idx, idx in enumerate(row_idxs):
                    item = items[idx]
                    dataset_name, img_name = item.split('___')
                    # generate origin and reconstruction
                    ori_img = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                      (dataset_name, img_name)), cv2.COLOR_BGR2RGB)
                    canvas.process_draw_image(ori_img, draw_idx, 0)
                    cur_code = test_batch['sean_code'][idx].copy()
                    cur_code[HAIR_IDX] = rec_code[idx].cpu().numpy()
                    rec_img = tensor2numpy(gen_by_sean(cur_code, item))
                    canvas.process_draw_image(rec_img, draw_idx, 1)

                for grid_idx in range(grid_count):
                    cur_noise_curliness = torch.tensor(lin_space[grid_idx]).reshape([1, 1]).tile([cfg.sample_batch_size, 1]).float()
                    data = copy.deepcopy(test_data)
                    data['noise_curliness'] = cur_noise_curliness
                    to_device(data, device)
                    code = gen(data)['code']
                    for draw_idx, idx in enumerate(row_idxs):
                        cur_code = test_batch['sean_code'][idx].copy()
                        cur_code[HAIR_IDX] = code[idx].cpu().numpy()
                        item = items[idx]
                        out_img = gen_by_sean(cur_code, item)
                        out_img = tensor2numpy(out_img)
                        canvas.process_draw_image(out_img, draw_idx, grid_idx + 2)
                if local_rank <= 0:
                    canvas.write_(os.path.join(save_dir, '%06d_curliness.png' % step))

            # ------------
            # generate variance
            # ------------
            if cfg.lambda_pca_std:
                grid_count = 10
                lin_space = np.linspace(10, 150, grid_count)
                canvas = Canvas(len(row_idxs), grid_count + 2)
                for draw_idx, idx in enumerate(row_idxs):
                    item = items[idx]
                    dataset_name, img_name = item.split('___')
                    # generate origin and reconstruction
                    ori_img = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                      (dataset_name, img_name)), cv2.COLOR_BGR2RGB)
                    canvas.process_draw_image(ori_img, draw_idx, 0)

                    cur_code = test_batch['sean_code'][idx].copy()
                    cur_code[HAIR_IDX] = rec_code[idx].cpu().numpy()
                    rec_img = tensor2numpy(gen_by_sean(cur_code, item))
                    canvas.process_draw_image(rec_img, draw_idx, 1)

                for grid_idx in range(grid_count):
                    cur_pca_std = torch.tensor(lin_space[grid_idx]).reshape([1, 1]).tile([cfg.sample_batch_size, 1]).float()
                    data = copy.deepcopy(test_data)
                    data['pca_std'] = cur_pca_std
                    to_device(data, device)
                    code = gen(data)['code']

                    for draw_idx, idx in enumerate(row_idxs):
                        cur_code = test_batch['sean_code'][idx].copy()
                        cur_code[HAIR_IDX] = code[idx].cpu().numpy()
                        item = items[idx]
                        out_img = gen_by_sean(cur_code, item)
                        out_img = tensor2numpy(out_img)
                        canvas.process_draw_image(out_img, draw_idx, grid_idx + 2)
                if local_rank <= 0:
                    canvas.write_(os.path.join(save_dir, '%06d_variance.png' % step))

            # -------------
            # generate each rgb
            # -------------
            canvas = Canvas(len(row_idxs) + 1, len(instance_idxs) + 1)
            for draw_idx, instance_idx in enumerate(instance_idxs):
                item = items[instance_idx]
                dataset_name, img_name = item.split('___')
                # generate origin and reconstruction
                ori_img = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                  (dataset_name, img_name)), cv2.COLOR_BGR2RGB)
                ori_img[5:45, 5:45, :] = test_batch['rgb_mean'][instance_idx].cpu().numpy()
                canvas.process_draw_image(ori_img, 0, draw_idx + 1)
            for draw_idx, idx in enumerate(row_idxs):
                item_row = items[idx]
                dataset_name, img_name = item_row.split('___')
                img_row = cv2.cvtColor(cv2.imread(os.path.join(cfg.data_root, '%s/images_256/%s.png') %
                                                  (dataset_name, img_name)), cv2.COLOR_BGR2RGB)
                canvas.process_draw_image(img_row, draw_idx + 1, 0)
            for draw_idx2, instance_idx in enumerate(instance_idxs):
                color = test_batch['rgb_mean'][[instance_idx]]
                data = copy.deepcopy(test_data)
                data['rgb_mean'] = torch.tile(color, [cfg.sample_batch_size, 1])
                data['pca_std'] = torch.tile(test_batch['pca_std'][[instance_idx]], [cfg.sample_batch_size, 1])
                to_device(data, device)
                hair_code = gen(data)['code']
                for draw_idx, idx in enumerate(row_idxs):
                    item_row = items[idx]
                    cur_code = test_batch['sean_code'][idx].copy()
                    cur_code[HAIR_IDX] = hair_code[idx].cpu().numpy()
                    res_img = gen_by_sean(cur_code, item_row)
                    res_img = tensor2numpy(res_img)
                    canvas.process_draw_image(res_img, draw_idx + 1, draw_idx2 + 1)
            if local_rank <= 0:
                canvas.write_(os.path.join(save_dir, '%06d_rgb.png' % step))

        gen.train()
        dis.train()
        if local_rank >= 0:
            dist.barrier()

    if step > 0 and step % cfg.model_save_step == 0:
        if local_rank <= 0:
            save_model(step, solver, ckpt_dir)
        if local_rank >= 0:
            dist.barrier()
