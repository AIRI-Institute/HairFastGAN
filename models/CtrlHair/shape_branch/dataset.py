# -*- coding: utf-8 -*-

"""
# File name:    dataset.py
# Time :        2021/11/16 21:24
# Author:       xyguoo@163.com
# Description:  
"""
import os

import cv2

from common_dataset import DataFilter
import random
import pickle
import torch
from torchvision import transforms as tform
from PIL import Image

from global_value_utils import GLOBAL_DATA_ROOT
from util.util import path_join_abs


class Dataset(DataFilter):

    def __init__(self, cfg, rank=0):
        super().__init__(cfg)
        self.cfg = cfg

        img_size = 256
        self.__setattr__('mask_transform_%d' % img_size,
                         tform.Compose([tform.Resize(img_size, interpolation=Image.NEAREST),
                                        tform.ToTensor(),
                                        tform.Lambda(lambda x: 255 * x)]))
        self.__setattr__('mask_transform_mirror_%d' % img_size,
                         tform.Compose([tform.Resize(img_size, interpolation=Image.NEAREST),
                                        tform.Lambda(lambda x: tform.functional.hflip(x)),
                                        tform.ToTensor(),
                                        tform.Lambda(lambda x: 255 * x)]))

        self.mask_pool_dir = os.path.join(cfg.data_root, cfg.adaptor_pool_dir)
        self.mask_test_pool_dir = os.path.join(cfg.data_root, cfg.adaptor_test_pool_dir)
        self.mask_buffer = []
        self.local_rank = rank
        random.seed(self.random_seed + self.local_rank + 1)

        if self.cfg.only_celeba_as_real:  # CelebA Mask is the manual mask, which has strong realism
            self.dis_real_list = [st for st in self.train_list if 'CelebaMask' in st]

        self.data_root = GLOBAL_DATA_ROOT
        with open(os.path.join(self.data_root, 'sean_code_dict.pkl'), 'rb') as f:
            self.sean_code_dict = pickle.load(f)

    def get_by_file_name(self, size, img_path=None, validate_func=None, mirror=False, data_list=None, need_img=False):
        if data_list is None:
            data_list = self.train_list

        if img_path is None:
            while True:
                random_idx = random.randint(0, len(data_list) - 1)
                img_path = data_list[random_idx]
                if validate_func is None or validate_func(path_join_abs(img_path, '../..'), img_path[-9:-4]):
                    break
                if data_list is self.train_list:
                    self.train_list = self.train_list[:random_idx] + self.train_list[random_idx + 1:]
                else:
                    data_list = data_list[:random_idx] + data_list[random_idx + 1:]

        if mirror:
            mask_transform = self.__getattribute__('mask_transform_mirror_%d' % size)
        else:
            mask_transform = self.__getattribute__('mask_transform_%d' % size)

        base_num = img_path[-9:-4]
        mask_path = path_join_abs(img_path, '../../label', base_num + '.png')
        mask = mask_transform(Image.open(mask_path))

        if need_img:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return mask, img
        return mask

    def get_pair_file_randomly_from_buffer(self):
        if len(self.mask_buffer) == 0:
            self.mask_buffer = os.listdir(self.mask_pool_dir)
            random.shuffle(self.mask_buffer)
        return self.mask_buffer.pop()

    def get_pair_randomly_from_pool(self, img_size):
        mask_file = self.get_pair_file_randomly_from_buffer()
        file_parts = mask_file.split('___')
        hair_dir, hair, face_dir, face = file_parts[:4]
        images_dir = 'label'
        mirror = (random.random() > 0.5)
        face_img, target_hair_mask, hair_img = self.get_adaptor_pair_mask(
            path_join_abs(self.cfg.data_root, face_dir, images_dir, '%s.png' % face),
            path_join_abs(self.cfg.data_root, hair_dir, images_dir, '%s.png' % hair),
            img_size, self.mask_pool_dir, mirror=mirror, mask_thread_num=file_parts[-1][:2])
        return face_img, target_hair_mask, hair_img

    def get_adaptor_pair_mask(self, face_path, hair_path, img_size, mask_pool_dir, mirror=False, mask_thread_num='00',
                              need_img=False):
        if need_img:
            face_mask, face_img = self.get_by_file_name(img_size, face_path, mirror=mirror, need_img=need_img)
            hair_mask, hair_img = self.get_by_file_name(img_size, hair_path, mirror=mirror, need_img=need_img)
        else:
            face_mask = self.get_by_file_name(img_size, face_path, mirror=mirror, need_img=need_img)
            hair_mask = self.get_by_file_name(img_size, hair_path, mirror=mirror, need_img=need_img)
        face_base_path = os.path.basename(face_path)[:-4]
        hair_base_path = os.path.basename(hair_path)[:-4]
        face_dir = face_path.split('/')[-3]
        hair_dir = hair_path.split('/')[-3]
        target_mask_path = os.path.join(mask_pool_dir, '%s___%s___%s___%s___%s.png' % (
            hair_dir, hair_base_path, face_dir, face_base_path, mask_thread_num))
        if mirror:
            transform_func = self.__getattribute__('mask_transform_mirror_%d' % img_size)
        else:
            transform_func = self.__getattribute__('mask_transform_%d' % img_size)

        target_hair_mask = transform_func(Image.open(target_mask_path))
        if need_img:
            return face_mask, target_hair_mask, hair_mask, face_img, hair_img
        else:
            return face_mask, target_hair_mask, hair_mask

    def get_random_pair_batch(self, batch_size, img_size=None):
        if not img_size:
            img_size = self.cfg.img_size
        face_imgs, target_hair_masks, hair_imgs = [], [], []
        while len(face_imgs) < batch_size:
            face_img, target_hair_mask, hair_img = \
                self.get_pair_randomly_from_pool(img_size)
            face_imgs.append(face_img)
            target_hair_masks.append(target_hair_mask)
            hair_imgs.append(hair_img)
        results = [face_imgs, target_hair_masks, hair_imgs]
        for idx in range(len(results)):
            results[idx] = torch.stack(results[idx], dim=0)
        return {'face': results[0], 'target': results[1], 'hair': results[2]}

    def get_random_single_batch(self, batch_size):
        face_imgs = []
        while len(face_imgs) < batch_size:
            if self.cfg.only_celeba_as_real:
                face_img = self.get_by_file_name(256, validate_func=self.valid_hair, mirror=(random.random() > 0.5),
                                                 data_list=self.dis_real_list)
            else:
                face_img = self.get_by_file_name(256, validate_func=self.valid_hair, mirror=(random.random() > 0.5))
            face_imgs.append(face_img)
        face_imgs = torch.stack(face_imgs, dim=0)
        return face_imgs

    def get_test_batch(self, batch_size=32, img_size=None):
        if not img_size:
            img_size = self.cfg.img_size
        face_masks, target_hair_masks, hair_masks, sean_codes, face_imgs, hair_imgs = [], [], [], [], [], []

        idx = 0
        while len(face_masks) < batch_size:
            face_maks_path = self.test_face_list[idx]
            hair_mask_path = self.test_hair_list[idx]
            idx += 1
            face_mask, target_hair_mask, hair_mask, face_img, hair_img = \
                self.get_adaptor_pair_mask(face_maks_path, hair_mask_path, img_size, self.mask_test_pool_dir,
                                           need_img=True)
            face_masks.append(face_mask)
            target_hair_masks.append(target_hair_mask)
            hair_masks.append(hair_mask)

            face_path_parts = face_maks_path.split('/')
            sean_codes.append(torch.tensor(
                self.sean_code_dict['___'.join([face_path_parts[-3], face_path_parts[-1][:-4]])]))

            face_imgs.append(torch.tensor(face_img))
            hair_imgs.append(torch.tensor(hair_img))

        face_masks = torch.stack(face_masks, dim=0)
        hair_masks = torch.stack(hair_masks, dim=0)
        target_hair_masks = torch.stack(target_hair_masks, dim=0)

        return {'face': face_masks, 'target': target_hair_masks, 'hair': hair_masks, 'sean_code': sean_codes,
                'face_imgs': face_imgs, 'hair_imgs': hair_imgs}


if __name__ == '__main__':
    from shape_branch.config import cfg

    ds = Dataset(cfg)
    # resources = ds.get_training_batch(8)
    res = ds.get_random_inpainting_batch(9)
    pass
