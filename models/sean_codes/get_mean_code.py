# -*- coding: utf-8 -*-

"""
# File name:    get_mean_code.py
# Time :        2022/2/22 17:22
# Author:       xyguoo@163.com
# Description:  
"""

from glob import glob

import numpy as np
import os

layers_list = ['ACE.npy']

style_list = []

for cat_i in range(19):
    for layer_j in layers_list:
        tmp_list = glob('styles_test/style_codes/*/' + str(cat_i) + '/' + layer_j)
        style_list = []

        for k in tmp_list:
            style_list.append(np.load(k))

        if len(style_list) > 0:
            style_list = np.array(style_list)

            style_list_norm2 = np.linalg.norm(style_list, axis=1, keepdims=True) ** 2
            dist_matrix = (style_list_norm2 + style_list_norm2.T -2 * style_list @ style_list.T)
            dist_matrix[dist_matrix < 0] = 0
            dist_matrix = dist_matrix ** 0.5
            median_idx = dist_matrix.sum(axis=1).argmin()
            feature = style_list[median_idx]

            save_folder = os.path.join('styles_test/mean_style_code/median', str(cat_i))

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_name = os.path.join(save_folder, layer_j)
            np.save(save_name, feature)

print(100)
