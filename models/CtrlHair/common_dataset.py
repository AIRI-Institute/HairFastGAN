# -*- coding: utf-8 -*-

"""
# File name:    common_dataset.py
# Time :        2021/12/31 19:19
# Author:       xyguoo@163.com
# Description:  
"""
import os
import random

import numpy as np
import pandas as pd

from global_value_utils import GLOBAL_DATA_ROOT, HAIR_IDX, HAT_IDX, DATASET_NAME
from util.util import path_join_abs
import cv2


class DataFilter:
    def __init__(self, cfg=None, test_part=0.096):
        # cfg
        if cfg is None:
            from color_texture_branch.config import cfg
        self.cfg = cfg

        base_dataset_dir = GLOBAL_DATA_ROOT

        dataset_dir = [os.path.join(base_dataset_dir, dn) for dn in DATASET_NAME]

        self.data_dirs = dataset_dir

        self.random_seed = 7
        random.seed(self.random_seed)

        ####################################################################
        # Please modify these if you don't want to use these filters
        angle_filter = True
        gender_filter = True
        gender = ['female']
        # gender = ['male', 'female']
        gender = set([{'male': 1, 'female': -1}[g] for g in gender])
        ####################################################################

        self.total_list = []
        for data_dir in self.data_dirs:
            img_dir = os.path.join(data_dir, 'images_256')

            if angle_filter:
                angle_csv = pd.read_csv(os.path.join(data_dir, 'angle.csv'), index_col=0)
                angle_filter_imgs = list(angle_csv.index[angle_csv['angle'] < 5])
                cur_list = ['%05d.png' % dd for dd in angle_filter_imgs]
            else:
                cur_list = os.listdir(img_dir)

            if gender_filter:
                attr_filter = pd.read_csv(os.path.join(data_dir, 'attr_gender.csv'))
                cur_list = [p for p in cur_list if attr_filter.Male[int(p[:-4])] in gender]

            self.total_list += [os.path.join(img_dir, p) for p in cur_list]

        random.shuffle(self.total_list)
        self.test_start = int(len(self.total_list) * (1 - test_part))
        self.test_list = self.total_list[self.test_start:]
        self.train_list = [st for st in self.total_list if st not in self.test_list]

        idx = 0
        # make sure the area of hair is big enough
        self.hair_region_threshold = 0.07
        self.test_face_list = []
        while len(self.test_face_list) < cfg.sample_batch_size:
            test_file = self.test_list[idx]
            if self.valid_face(path_join_abs(test_file, '../..'), test_file[-9:-4]):
                self.test_face_list.append(test_file)
            idx += 1

        self.test_hair_list = []
        while len(self.test_hair_list) < cfg.sample_batch_size:
            test_file = self.test_list[idx]
            if self.valid_hair(path_join_abs(test_file, '../..'), test_file[-9:-4]):
                self.test_hair_list.append(test_file)
            idx += 1

    def valid_face(self, data_dir, img_idx_str):
        label_path = os.path.join(data_dir, 'label', img_idx_str + '.png')
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        hat_region = label_img == HAT_IDX

        if hat_region.mean() > 0.03:
            return False
        return True

    def valid_hair(self, data_dir, img_idx_str):
        label_path = os.path.join(data_dir, 'label', img_idx_str + '.png')
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        hair_region = label_img == HAIR_IDX
        hat_region = label_img == HAT_IDX

        if hat_region.mean() > 0.03:
            return False
        if hair_region.mean() < self.hair_region_threshold:
            return False
        return True

