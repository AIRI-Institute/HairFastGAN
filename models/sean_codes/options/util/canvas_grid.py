# -*- coding: utf-8 -*-

"""
# File name:    canvas_grid.py
# Time :        2021/11/18 16:40
# Author:       xyguoo@163.com
# Description:  This is the util that generating image in a big grid picture
"""

import numpy as np

from util.imutil import write_rgb


class Canvas:
    def __init__(self, row, col, img_size=256, margin=0):
        self.row = row
        self.col = col
        self.img_size = img_size
        self.margin = margin
        self.canvas = np.ones((row * img_size, col * img_size + margin * (col - 1), 3), dtype='uint8') * 255

    def process_draw_image(self, img, i, j):
        if img.dtype in [np.float32, np.float, np.float64]:
            if img.min() < 0:
                img = img * 127.5 + 127.5
            elif img.max() <= 1:
                img = img * 255
            img = img.astype('uint8')
        i_start, j_start = int(i * self.img_size), int(j * self.img_size) + int(j * self.margin)
        self.canvas[i_start: i_start + img.shape[0], j_start: j_start + img.shape[1], :] = img

    def write_(self, file):
        write_rgb(file, self.canvas)
