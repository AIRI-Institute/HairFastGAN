# -*- coding: utf-8 -*-

"""
# File name:    color_from_hsv_to_gaussian.py
# Time :        2022/3/6 16:00
# Author:       xyguoo@163.com
# Description:  transfer HSV value to Gaussian latent value according to the distribution of the dataset
"""
import os
import pickle as pkl
from bisect import bisect_left, bisect_right

import scipy.stats as st


class DistTranslation:
    def __init__(self):
        hair_root = 'dataset_info_ctrlhair'
        with open(os.path.join(hair_root, 'hsv_stat_dict_ordered.pkl'), 'rb') as f:
            self.cols_hsv = pkl.load(f)

    def gaussian_to_val(self, dim, val):
        # if dim == 0:
        #     return (val + 2.) / 4 * 179
        return self.cols_hsv[int((st.norm.cdf(val)) * self.cols_hsv.shape[0])][dim]

    def val_to_gaussian(self, dim, val):
        # if dim == 0:
        #     return val / 179 * 2 * 2. - 2.

        left_v = bisect_left(self.cols_hsv[:, dim], val)
        right_v = bisect_right(self.cols_hsv[:, dim], val)
        return st.norm.ppf((left_v + right_v) / 2 / self.cols_hsv.shape[0])

#
# if __name__ == '__main__':
#     dt = DistTranslation()
#
#     with open('hsv_stat_dict_ordered.pkl', 'wb') as f:
#         pkl.dump(dt.cols_hsv, f)
