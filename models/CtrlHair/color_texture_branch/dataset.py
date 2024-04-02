# -*- coding: utf-8 -*-

"""
# File name:    dataset.py
# Time :        2021/11/16 21:24
# Author:       xyguoo@163.com
# Description:  
"""
import os

from common_dataset import DataFilter

import random
import numpy as np
from global_value_utils import HAIR_IDX, GLOBAL_DATA_ROOT
import pickle
import torch


class Dataset:
    """A general image-attributes dataset class."""

    def valid_hair(self, item):
        if np.isnan(self.rgb_stat_dict[item][0]).any():  # not have hair
            return False
        if (self.sean_code_dict[item][HAIR_IDX] == 0).all():
            return False
        if item not in self.color_var_stat_dict:
            return False
        return True

    def __init__(self, cfg, rank=0, test_part=0.096):
        # super().__init__()
        self.cfg = cfg

        self.random_seed = 7  # Do not change the random seed, which determines the split of scripts and test set
        self.hair_root = GLOBAL_DATA_ROOT
        with open(os.path.join(self.hair_root, 'sean_code_dict.pkl'), 'rb') as f:
            self.sean_code_dict = pickle.load(f)
        with open(os.path.join(self.hair_root, 'rgb_stat_dict.pkl'), 'rb') as f:
            self.rgb_stat_dict = pickle.load(f)
        with open(os.path.join(self.hair_root, 'color_var_stat_dict.pkl'), 'rb') as f:
            self.color_var_stat_dict = pickle.load(f)

        self.local_rank = rank
        random.seed(self.random_seed + self.local_rank + 1)

        self.data_list = [dd for dd in list(self.sean_code_dict) if self.valid_hair(dd)]
        random.shuffle(self.data_list)

        self.data_filter = DataFilter(cfg, test_part)

        self.test_list = []
        for ll in self.data_filter.test_list:
            path_part = ll.split('/')
            self.test_list.append('%s___%s' % (path_part[-3], path_part[-1][:-4]))

        self.train_filter = []
        for ll in self.data_filter.train_list:
            path_part = ll.split('/')
            self.train_filter.append('%s___%s' % (path_part[-3], path_part[-1][:-4]))
        self.train_filter = set(self.train_filter)
        self.train_list = [ll for ll in self.data_list if ll not in self.test_list]
        if cfg.filter_female_and_frontal:
            self.train_list = [ll for ll in self.train_list if ll in self.train_filter]
        self.train_set = set(self.train_list)

        ### curliness code
        self.curliness_hair_list = {}
        self.curliness_hair_list_test = {}
        self.curliness_hair_dict = {ke: 0 for ke in self.color_var_stat_dict}

        for label in [-1, 1]:
            img_file = os.path.join(cfg.data_root, 'manual_label', 'curliness', '%d.txt' % label)
            with open(img_file, 'r') as f:
                imgs = [l.strip() for l in f.readlines()]
            imgs = [ii for ii in imgs if ii in self.train_set]
            self.curliness_hair_list[label] = imgs
            for ii in imgs:
                self.curliness_hair_dict[ii] = label

            img_file = os.path.join(cfg.data_root, 'manual_label', 'curliness', 'test_%d.txt' % label)
            with open(img_file, 'r') as f:
                imgs = [l.strip() for l in f.readlines()]
            self.curliness_hair_list_test[label] = imgs
            for ii in imgs:
                self.curliness_hair_dict[ii] = label

    def get_sean_code(self, ke):
        return self.sean_code_dict[ke]

    def get_list_by_items(self, items):
        res_code, res_rgb_mean, res_pca_std, res_sean_code, res_curliness = [], [], [], [], []
        for item in items:
            code = self.sean_code_dict[item][HAIR_IDX]
            res_code.append(code)
            rgb_mean = self.rgb_stat_dict[item][0]
            res_rgb_mean.append(rgb_mean)
            pca_std = self.color_var_stat_dict[item]['var_pca']
            # here the 'var' is 'std'
            res_pca_std.append(pca_std[..., None])
            res_sean_code.append(self.get_sean_code(ke=item))
            res_curliness.append(self.curliness_hair_dict[item])
        res_code = torch.tensor(np.stack(res_code), dtype=torch.float32)
        res_rgb_mean = torch.tensor(np.stack(res_rgb_mean), dtype=torch.float32)
        res_pca_std = torch.tensor(np.stack(res_pca_std), dtype=torch.float32)
        res_curliness = torch.tensor(np.stack(res_curliness), dtype=torch.int)[..., None]
        data = {'code': res_code, 'rgb_mean': res_rgb_mean, 'pca_std': res_pca_std, 'items': items,
                'sean_code': res_sean_code, 'curliness_label': res_curliness}
        return data

    def get_training_batch(self, batch_size):
        items = []
        while len(items) < batch_size:
            item = random.choice(self.train_list)
            # if not self.valid_hair(item):
            #     continue
            items.append(item)
        data = self.get_list_by_items(items)
        return data

    def get_testing_batch(self, batch_size):
        ptr = 0
        items = []
        while len(items) < batch_size:
            item = self.test_list[ptr]
            ptr += 1
            if not self.valid_hair(item):
                continue
            items.append(item)
        data = self.get_list_by_items(items)
        return data

    def get_curliness_hair(self, labels):
        labels = labels.cpu().numpy()
        items = []
        for label in labels:
            item_list = self.curliness_hair_list[label[0]]
            items.append(np.random.choice(item_list))
        data = self.get_list_by_items(items)
        return data

    def get_curliness_hair_test(self):
        return self.get_list_by_items(self.curliness_hair_list_test[-1] + self.curliness_hair_list_test[1])


# if __name__ == '__main__':
#     ds = Dataset(cfg)
#     resources = ds.get_training_batch(8)
#     pass


# for label in [-1, 1]:
#     img_dir = os.path.join(cfg.data_root, 'manual_label', 'curliness', 'test_%d' % label)
#     imgs = [pat[:-4] + '\n' for pat in os.listdir(img_dir)]
#     imgs.sort()
#     with open(img_dir + '.txt', 'w') as f:
#         f.writelines(imgs)
