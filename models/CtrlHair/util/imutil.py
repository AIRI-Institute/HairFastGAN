# -*- coding: utf-8 -*-

"""
# File name:    imutil.py
# Time :        2021/12/7 14:55
# Author:       xyguoo@163.com
# Description:  
"""
import cv2
import numpy as np


def read_rgb(img_path):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def write_rgb(file_name, img):
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=2)
    elif img.shape[2] == 1:
        img = np.tile(img, [1, 1, 3])
    cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
