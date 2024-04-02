# -*- coding: utf-8 -*-

"""
# File name:    global_value_utils.py
# Time :        2021/9/11 19:15
# Author:       xyguoo@163.com
# Description:  
"""
import os
import pickle as pkl


GLOBAL_DATA_ROOT = './dataset_info_ctrlhair'
########################################################
# Please change these setting if you scripts a new model`
DATASET_NAME = ['ffhq', 'CelebaMask_HQ']
DEFAULT_CONFIG_COLOR_TEXTURE_BRANCH = '045'
DEFAULT_CONFIG_SHAPE_BRANCH = '054'
########################################################

TEMP_FOLDER = 'models/CtrlHair/temp_folder'

PARSING_COLOR_LIST = [[0, 0, 0],
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
                      [0, 0, 204],  ## hair
                      [255, 51, 153],  ##
                      [0, 204, 204],  ##
                      [0, 51, 0],  ##
                      [255, 153, 51],
                      [0, 204, 0],
                      [255, 85, 255],
                      [255, 170, 255],
                      [0, 170, 255],
                      [85, 255, 255],
                      [170, 255, 255],
                      [255, 255, 255]]

PARSING_LABEL_LIST = ['background', 'skin_other', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                      'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat',
                      'ear_r', 'neck_l', 'neck', 'cloth']
HAIR_IDX = PARSING_LABEL_LIST.index('hair')
HAT_IDX = PARSING_LABEL_LIST.index('hat')
UNKNOWN_IDX = len(PARSING_COLOR_LIST) - 1

WRAP_TEMP_FOLDER = [os.path.join(TEMP_FOLDER, '/wrap_triangle/wrap_temp_result')]
