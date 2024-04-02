# -*- coding: utf-8 -*-

"""
# File name:    adaptor_generation.py
# Time :        2021/12/31 19:35
# Author:       xyguoo@163.com
# Description:  
"""

import sys
import os

pp = os.path.abspath(os.path.join(os.path.abspath(__file__), '../..'))
sys.path.append(pp)

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from wrap_codes.mask_adaptor import wrap_for_image_with_idx
from common_dataset import DataFilter
from shape_branch.config import cfg
from global_value_utils import HAIR_IDX, TEMP_FOLDER
from util.util import path_join_abs
import cv2
import threading
import random
import time
import glob
import tqdm

def generate_mask_to_pool(root_dir, hair_dir, face_dir, hair_base_path, face_base_path, pool_dir, temp_wrap_dir,
                          thread_id, find_repeat, only_hair):
    face_num = face_base_path[-9:-4]
    hair_num = hair_base_path[-9:-4]
    output_name = os.path.join(pool_dir, '%s___%s___%s___%s___%02d.png') % (
        hair_dir, hair_num, face_dir, face_num, thread_id)
    find_res = []
    if find_repeat:
        find_res = glob.glob(os.path.join(pool_dir, '%s___%s___%s___%s___*.png') % (
            hair_dir, hair_num, face_dir, face_num))
    if len(find_res) == 0:
        print('Generate thread %d: %s %s to %s %s' % (thread_id, hair_dir, hair_num, face_dir, face_num))
        align_mask = wrap_for_image_with_idx(root_dir, hair_dir, face_dir,
                                             hair_base_path, face_base_path,
                                             wrap_temp_folder=temp_wrap_dir)[0]
        if only_hair:
            output_img = (align_mask == HAIR_IDX) * 255
        else:
            output_img = align_mask
        cv2.imwrite(output_name, output_img)
    else:
        print('Hit for %s' % output_name)


class AdaptorPoolGeneration:
    def __init__(self, only_hair, dir_name, test_dir_name, thread_num=10, max_file=1e7):
        self.data_filter = DataFilter(cfg)
        self.pool_dir = os.path.join(cfg.data_root, dir_name)
        self.pool_test_dir = os.path.join(cfg.data_root, test_dir_name)
        self.max_file = max_file
        self.only_hair = only_hair
        for p in [self.pool_dir, self.pool_test_dir]:
            if not os.path.exists(p):
                os.makedirs(p)
        self.thread_num = thread_num

    def generate_test_set(self, img_num=100):
        temp_wrap_dir = os.path.join(TEMP_FOLDER, 'wrap_triangle/temp_wrap_test')

        for hair in tqdm.tqdm(self.data_filter.test_hair_list[:img_num]):
            for face in self.data_filter.test_face_list[:img_num]:
                hair_dir = hair.split('/')[-3]
                face_dir = face.split('/')[-3]
                base_hair = os.path.split(hair)[-1]
                base_face = os.path.split(face)[-1]
                generate_mask_to_pool(cfg.data_root, hair_dir, face_dir, base_hair, base_face,
                                      self.pool_test_dir, temp_wrap_dir, 0, find_repeat=False, only_hair=self.only_hair)

    def run(self):
        self.threads = []
        for idx in range(self.thread_num):
            t = threading.Thread(target=self.generate_thread, args=[idx])
            self.threads.append(t)
        for thread in self.threads:
            thread.start()

    def generate_thread(self, thread_idx):
        temp_wrap_dir = os.path.join(TEMP_FOLDER, 'wrap_triangle/temp_wrap_%d' % thread_idx)
        random.seed(time.time())
        while True:
            if len(os.listdir(self.pool_dir)) < self.max_file:
                for _ in range(100):
                    while True:
                        hair_path = random.choice(self.data_filter.train_list)
                        hair_num = hair_path[-9:-4]
                        hair_dir = hair_path.split('/')[-3]
                        if self.data_filter.valid_hair(path_join_abs(hair_path, '../..'), hair_num):
                            break
                    while True:
                        face_path = random.choice(self.data_filter.train_list)
                        face_num = face_path[-9:-4]
                        face_dir = face_path.split('/')[-3]
                        if self.data_filter.valid_face(path_join_abs(face_path, '../..'), face_num):
                            break
                    try:
                        generate_mask_to_pool(cfg.data_root, hair_dir, face_dir,
                                              os.path.basename(hair_path), os.path.basename(face_path),
                                              self.pool_dir, temp_wrap_dir,
                                              thread_idx, find_repeat=False, only_hair=self.only_hair)
                    except Exception as e:
                        print(repr(e))
            else:
                print('Full, so sleep in thread %d' % thread_idx)
                time.sleep(3.0)
