# -*- coding: utf-8 -*-

"""
# File name:    script_get_rgb_hsv_label.py
# Time :        2021/11/16 21:23
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
import numpy as np
from global_value_utils import HAIR_IDX, GLOBAL_DATA_ROOT, DATASET_NAME

data_name = DATASET_NAME

root_dir = GLOBAL_DATA_ROOT
imgs_sub_dir = 'images_256'
target_dir = os.path.join(root_dir, 'hair_info_all_dataset/rgb_stat')

ds = DataFilter()
ds.total_list.sort()

path_list = []
for d in data_name:
    data_dir = os.path.join(root_dir, d, imgs_sub_dir)
    path_list += [os.path.join(data_dir, pp) for pp in os.listdir(data_dir)]

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print('Building RGB dict...')
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

    moment1 = points.mean(axis=0)
    moment2 = ((points - moment1) ** 2).mean(axis=0)
    moment3 = ((points - moment1) ** 3).mean(axis=0)
    moment4 = ((points - moment1) ** 4).mean(axis=0)

    target_file = os.path.join(target_dir, '%s___%s.pkl' % (dataset_name, base_name[:-4]))
    with open(target_file, 'wb') as f:
        pickle.dump([moment1, moment2, moment3, moment4], f)

rgb_stat_path = os.path.join(root_dir, 'rgb_stat_dict.pkl')
merge_pickle_dir_to_dict(target_dir, os.path.join(root_dir, 'rgb_stat_dict.pkl'))

### get hsv dict
print('Building ordered HSV dict for assisting distribution fitting...')
data_root = GLOBAL_DATA_ROOT
output_path = os.path.join(GLOBAL_DATA_ROOT, 'hsv_stat_dict_ordered.pkl')

with open(rgb_stat_path, 'rb') as f:
    rgb_stat_dict = pickle.load(f)

files = list(rgb_stat_dict)

cols = [rgb_stat_dict[f][0] for f in files]
cols = np.array(cols)
cols_hsv = cv2.cvtColor(cols[None, ...].astype('uint8'), cv2.COLOR_RGB2HSV)[0]

for dim in range(3):
    cols_hsv[:, dim].sort()

with open(output_path, 'wb') as f:
    pickle.dump(cols_hsv, f)
