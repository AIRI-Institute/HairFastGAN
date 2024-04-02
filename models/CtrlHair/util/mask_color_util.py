# -*- coding: utf-8 -*-

"""
# File name:    mask_color_util.py
# Time :        2021/12/7 11:13
# Author:       xyguoo@163.com
# Description:  
"""

import numpy as np

from models.CtrlHair.global_value_utils import HAIR_IDX


def mask_to_rgb(pred, draw_type=2):
    """
    generate visual mask image
    :param pred: pred is the label image with pixel vale in {0, 1, 2, ..., 20}
    :param draw_type: 0: all part;  1: {bg, face, hair};  2: {hair, others}
    :return:
    """
    if len(pred.shape) == 3 and pred.shape[0] == 1:
        pred = pred[0]
    num_labels = 19
    color = np.array([[0, 128, 64],
                      [204, 0, 0],
                      [76, 153, 0],
                      [204, 204, 0],  ##
                      [51, 51, 255],  ##
                      [204, 0, 204],  ##
                      [0, 255, 255],  ##
                      [51, 255, 255],  ##
                      [102, 51, 0],  ##
                      [255, 0, 0],  ##
                      [102, 204, 0],  ##
                      [255, 255, 0],  ##
                      [0, 0, 153],  ##
                      [0, 0, 204],  ##
                      [255, 51, 153],  ##
                      [0, 204, 204],  ##
                      [0, 51, 0],  ##
                      [255, 153, 51],
                      [0, 204, 0],
                      ])

    for cc in range(len(color)):
        if draw_type == 2:
            if cc != HAIR_IDX:
                color[cc] = [255, 255, 255]
        elif draw_type == 1:
            if cc != HAIR_IDX and cc != 0:
                color[cc] = [237, 28, 36]

    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = 255
    return rgb
