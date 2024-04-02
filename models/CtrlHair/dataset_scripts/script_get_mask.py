#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.append('.')

import tqdm

import os
import os.path as osp
import numpy as np
import cv2
from global_value_utils import GLOBAL_DATA_ROOT, PARSING_COLOR_LIST, DATASET_NAME
from util.imutil import read_rgb, write_rgb
from external_code.face_parsing.my_parsing_util import FaceParsing


data_name = [d for d in DATASET_NAME if d != 'CelebaMask_HQ']

def makedir(pat):
    if not os.path.exists(pat):
        os.makedirs(pat)


def vis_parsing_maps(im, parsing_anno, stride, save_im, save_path, img_path):
    # Colors for all 20 parts

    label_path = os.path.join(save_path, 'label')
    vis_path = os.path.join(save_path, 'vis')
    makedir(pat=label_path)
    makedir(pat=vis_path)

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    img_path = img_path[:-4] + '.png'

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        if len(index[0]) > 0:
            vis_parsing_anno_color[index[0], index[1], :] = PARSING_COLOR_LIST[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(vis_im, 0.4, vis_parsing_anno_color, 0.6, 0)

    cv2.imwrite(os.path.join(label_path, img_path), vis_parsing_anno)
    write_rgb(os.path.join(vis_path, img_path), vis_im)


def evaluate(respth, dspth):
    if not os.path.exists(respth):
        os.makedirs(respth)

    files = os.listdir(dspth)
    files.sort()
    for image_path in tqdm.tqdm(files):
        parsing, image = FaceParsing.parsing_img(read_rgb(osp.join(dspth, image_path)))
        parsing = FaceParsing.swap_parsing_label_to_celeba_mask(parsing)
        vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=respth, img_path=image_path)


if __name__ == "__main__":
    for dn in data_name:
        input_dir = os.path.join(GLOBAL_DATA_ROOT, dn, 'images_256')
        target_root_dir = os.path.join(GLOBAL_DATA_ROOT, dn)
        evaluate(respth=target_root_dir, dspth=input_dir)
