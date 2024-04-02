# -*- coding: utf-8 -*-

"""
# File name:    validation_in_train.py
# Time :        2021/12/10 12:55
# Author:       xyguoo@163.com
# Description:  
"""
import os

import copy

import cv2

from hair_editor import HairEditor
from my_torchlib.train_utils import generate_noise
from util.canvas_grid import Canvas
from util.imutil import write_rgb
from util.mask_color_util import mask_to_rgb
from .config import cfg
import my_pylib
import torch
import torch.distributed as dist
import my_torchlib
from .shape_util import split_hair_face, mask_one_hot_to_label
import numpy as np

he = HairEditor(load_feature_model=False, load_mask_model=False)


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

        gen.eval()
        # dis.eval()

        with torch.no_grad():
            target_hair_part, _ = split_hair_face(test_batch['target'])
            face_hair_part, face_part = split_hair_face(test_batch['face'])
            hair_hair_part, hair_face_part = split_hair_face(test_batch['hair'])

            face_face_code = solver.gen.forward_face_encoder(face_part)
            _, target_hair_code, _ = solver.gen.forward_hair_encoder(target_hair_part)
            hair_face_code = solver.gen.forward_face_encoder(hair_face_part)
            _, hair_hair_code, _ = solver.gen.forward_hair_encoder(hair_hair_part)

            hair_mask_label = mask_one_hot_to_label(test_batch['hair']).cpu().numpy()
            hair_mask_vis = [mask_to_rgb(ii, draw_type=0) for ii in hair_mask_label]

            face_mask_label = mask_one_hot_to_label(test_batch['face']).cpu().numpy()
            face_mask_vis = [mask_to_rgb(ii, draw_type=0) for ii in face_mask_label]

            target_mask_label = mask_one_hot_to_label(test_batch['target']).cpu().numpy()
            target_mask_vis = [mask_to_rgb(ii, draw_type=0) for ii in target_mask_label]

            # ---------
            # rec and edit random
            # ---------
            grid_count = 10
            # generate each noise dim
            canvas = Canvas(cfg.sample_batch_size, grid_count + 2, margin=3)

            rec_masks = solver.gen.forward_decode_by_code(hair_hair_code, hair_face_code)
            rec_masks_label = mask_one_hot_to_label(rec_masks).cpu().numpy()
            rec_masks_vis = [mask_to_rgb(ii, draw_type=0) for ii in rec_masks_label]
            for draw_idx, idx in enumerate(range(cfg.sample_batch_size)):
                canvas.process_draw_image(hair_mask_vis[draw_idx], draw_idx, 0)
                canvas.process_draw_image(rec_masks_vis[draw_idx], draw_idx, 1)

            temp_noise = generate_noise(grid_count, cfg.hair_dim).to(device)
            for grid_idx in range(grid_count):
                cur_hair_code = torch.tile(temp_noise[grid_idx, :], (cfg.sample_batch_size, 1))
                res_masks = solver.gen.forward_decode_by_code(cur_hair_code, hair_face_code)
                res_masks_label = mask_one_hot_to_label(res_masks).cpu().numpy()
                for draw_idx, idx in enumerate(range(cfg.sample_batch_size)):
                    canvas.process_draw_image(mask_to_rgb(res_masks_label[draw_idx], draw_type=0),
                                              draw_idx, grid_idx + 2)

            if local_rank <= 0:
                canvas.write_(os.path.join(out_dir, '%07d_random.png' % step))

            # ---------
            # transfer
            # ---------
            res_mask = solver.gen.forward_decode_by_code(target_hair_code, face_face_code)
            res_mask_label = mask_one_hot_to_label(res_mask).cpu().numpy()
            res_mask_vis = [mask_to_rgb(ii, draw_type=0) for ii in res_mask_label]
            vacant_column = np.ones([cfg.sample_batch_size * cfg.img_size, 3, 3], dtype='uint8') * 255

            face_imgs = np.concatenate(test_batch['face_imgs'], axis=0)
            hair_imgs = np.concatenate(test_batch['hair_imgs'], axis=0)
            target_imgs = []
            for idx in range(cfg.sample_batch_size):
                cur_img = he.gen_img(test_batch['sean_code'][idx][None, ...],
                                     res_mask_label[idx][None, None, ...]).cpu().numpy() * 127.5 + 127.5
                cur_img = np.transpose(cur_img, [1, 2, 0])
                cur_img, _ = he.postprocess_blending(test_batch['face_imgs'][idx], cur_img,
                                                     face_mask_label[idx][None, ...],
                                                     res_mask_label[idx][None, ...])
                target_imgs.append(cur_img)
            target_imgs = np.concatenate(target_imgs, axis=0)
            canvas = np.concatenate([np.concatenate(hair_mask_vis, axis=0), vacant_column,
                                     hair_imgs, vacant_column,
                                     np.concatenate(face_mask_vis, axis=0), vacant_column,
                                     face_imgs, vacant_column,
                                     np.concatenate(target_mask_vis, axis=0), vacant_column,
                                     np.concatenate(res_mask_vis, axis=0), vacant_column,
                                     target_imgs], axis=1)
            write_rgb(os.path.join(out_dir, '%07d__transfer.png' % step), canvas)

            # ---------
            # edit code
            # ---------
            grid_count = 6
            lin_space = np.linspace(-2.5, 2.5, grid_count)

            for dim_idx in range(cfg.hair_dim):
                canvas = Canvas(cfg.sample_batch_size, grid_count + 1, margin=3)

                for draw_idx, idx in enumerate(range(cfg.sample_batch_size)):
                    canvas.process_draw_image(hair_mask_vis[draw_idx], draw_idx, 0)

                for grid_idx in range(grid_count):
                    hair_hair_code_copy = copy.deepcopy(hair_hair_code)
                    hair_hair_code_copy[:, dim_idx] = lin_space[grid_idx]
                    res_mask = solver.gen.forward_decode_by_code(hair_hair_code_copy, hair_face_code)
                    res_mask = mask_one_hot_to_label(res_mask).cpu().numpy()

                    for draw_idx, idx in enumerate(range(cfg.sample_batch_size)):
                        draw_mask = mask_to_rgb(res_mask[draw_idx], draw_type=0)
                        canvas.process_draw_image(draw_mask, draw_idx, grid_idx + 1)
                if local_rank <= 0:
                    canvas.write_(os.path.join(out_dir, '%07d_noise_%02d.png' % (step, dim_idx)))

        gen.train()
        # dis.scripts()
        if local_rank >= 0:
            dist.barrier()

    if step > 0 and step % cfg.model_save_step == 0:
        if local_rank <= 0:
            save_model(step, solver, ckpt_dir)
        if local_rank >= 0:
            dist.barrier()
