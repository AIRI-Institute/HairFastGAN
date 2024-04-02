# -*- coding: utf-8 -*-

"""
# File name:    script_get_sean_code.py
# Time :        2021/11/16 15:56
# Author:       xyguoo@163.com
# Description:  
"""

import os
import sys
sys.path.append('.')

from dataset_scripts.utils import merge_pickle_dir_to_dict
import cv2
import tqdm
from global_value_utils import GLOBAL_DATA_ROOT, DATASET_NAME

from hair_editor import HairEditor
import pickle

data_name = DATASET_NAME

root_dir = GLOBAL_DATA_ROOT
imgs_sub_dir = 'images_256'
target_dir = os.path.join(root_dir, 'hair_info_all_dataset/sean_code')

he = HairEditor(load_mask_model=False)
path_list = []
for d in data_name:
    data_dir = os.path.join(root_dir, d, imgs_sub_dir)
    path_list += [os.path.join(data_dir, pp) for pp in os.listdir(data_dir)]

path_list.sort()
# res_dict = {}

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for img_path in tqdm.tqdm(path_list):
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
    # resize
    hair_img = he.preprocess_img(hair_img)
    hair_parsing = he.preprocess_mask(hair_parsing)
    cur_code = he.get_code(hair_img, hair_parsing)
    cur_code = cur_code.cpu().numpy()[0]
    # res_dict['%s___%s' % (dataset_name, base_name)] = cur_code

    target_file = os.path.join(target_dir, '%s___%s.pkl' % (dataset_name, base_name[:-4]))
    with open(target_file, 'wb') as f:
        pickle.dump(cur_code, f)

merge_pickle_dir_to_dict(target_dir, os.path.join(root_dir, 'sean_code_dict.pkl'))
