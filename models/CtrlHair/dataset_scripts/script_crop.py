# -*- coding: utf-8 -*-

"""
# File name:    script_crop.py
# Time :        2022/07/17
# Author:       xyguoo@163.com
# Description:  
"""

import sys
sys.path.append('.')

from external_code.crop import recreate_aligned_images
from global_value_utils import GLOBAL_DATA_ROOT

import os
from external_code.landmarks_util import predictor_dict, detector
import numpy as np
import cv2

predictor_68 = predictor_dict[68]

##############################################
# Please input your dataset dir
root_dir = 'your/dataset/original/images'
dataset_name = 'your_dataset_name'
##############################################

dataset_dir = os.path.join(GLOBAL_DATA_ROOT, dataset_name)
out_dir = os.path.join(dataset_dir, 'images_256')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

files = os.listdir(root_dir)
files.sort()
for face_path in files:
    face_img_bgr = cv2.imread(os.path.join(root_dir, face_path))
    face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    face_img_rgb = cv2.resize(face_img_rgb, dsize=(face_img_rgb.shape[1], face_img_rgb.shape[0]))
    face_bbox = detector(face_img_rgb, 0)
    face_lm_68 = np.array([[p.x, p.y] for p in predictor_68(face_img_bgr, face_bbox[0]).parts()])

    face_img_pil, _ = recreate_aligned_images(face_img_rgb, face_lm_68, output_size=256)

    img_np = np.array(face_img_pil)
    cv2.imwrite(os.path.join(out_dir, face_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))