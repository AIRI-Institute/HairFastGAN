# -*- coding: utf-8 -*-

"""
# File name:    script_get_color_var_label.py
# Time :        2021/11/29 22:10
# Author:       xyguoo@163.com
# Description:  
"""
import os
import sys
sys.path.append('.')

from dataset_scripts.utils import merge_pickle_dir_to_dict
import cv2
import tqdm

import pickle
from common_dataset import DataFilter
from global_value_utils import HAIR_IDX, GLOBAL_DATA_ROOT, DATASET_NAME
import numpy as np


data_name = DATASET_NAME

root_dir = GLOBAL_DATA_ROOT
imgs_sub_dir = 'images_256'
target_dir = os.path.join(root_dir, 'hair_info_all_dataset/color_var_stat')

ds = DataFilter()
ds.total_list.sort()

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

path_list = []
for d in data_name:
    data_dir = os.path.join(root_dir, d, imgs_sub_dir)
    path_list += [os.path.join(data_dir, pp) for pp in os.listdir(data_dir)]

res_dict = {}
for img_path in tqdm.tqdm(path_list[:], total=len(path_list)):
    for dd in data_name:
        if img_path.find(dd) != -1:
            dataset_name = dd
            break
    else:
        raise NotImplementedError
    base_name = os.path.basename(img_path)
    hair_path = os.path.join(root_dir, dataset_name, imgs_sub_dir, base_name)
    hair_img = cv2.cvtColor(cv2.imread(hair_path), cv2.COLOR_BGR2RGB)
    hair_parsing = cv2.imread(os.path.join(root_dir, dataset_name, 'label', base_name), cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(hair_parsing.astype('uint8'), hair_img.shape[:2], interpolation=cv2.INTER_NEAREST)

    hair_mask = (mask_img == HAIR_IDX).astype('uint8')
    kernel13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(19, 19))
    hair_mask = cv2.erode(hair_mask, kernel13, iterations=1)

    points = hair_img[hair_mask.astype('bool')]

    # points color
    color_var_stat = []
    var_rgb = (points / 255).var(axis=0)
    if len(points) > 5:
        points_hsv = cv2.cvtColor(points[None, ...], cv2.COLOR_RGB2HSV)
        points_hsv = points_hsv / np.array([180, 255, 255])
        var_hsv = points_hsv.var(axis=(0, 1))

        points_hls = cv2.cvtColor(points[None, ...], cv2.COLOR_RGB2HLS)
        points_hls = points_hls / np.array([180, 255, 255])
        var_hls = points_hls.var(axis=(0, 1))

        points_lab = cv2.cvtColor(points[None, ...], cv2.COLOR_RGB2LAB)
        points_lab = points_lab / np.array([255, 100, 100])
        var_lab = points_lab.var(axis=(0, 1))

        points_yuv = cv2.cvtColor(points[None, ...], cv2.COLOR_RGB2YUV)
        points_yuv = points_yuv / np.array([255, 200, 200])
        var_yuv = points_yuv.var(axis=(0, 1))
        color_var_stat = {'var_rgb': var_rgb, 'var_hsv': var_hsv, 'var_hls': var_hls,
                          'var_yuv': var_yuv}

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(points)
        pca_dim = pca.transform(points)[:, 0]
        color_var_stat['var_pca'] = pca_dim.std()
        color_var_stat['var_pca_mean'] = pca.mean_
        color_var_stat['var_pca_comp'] = pca.components_

        target_file = os.path.join(target_dir, '%s___%s.pkl' % (dataset_name, base_name[:-4]))
        with open(target_file, 'wb') as f:
            pickle.dump(color_var_stat, f)

merge_pickle_dir_to_dict(target_dir, os.path.join(root_dir, 'color_var_stat_dict.pkl'))