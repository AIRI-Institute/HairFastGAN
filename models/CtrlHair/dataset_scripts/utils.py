# -*- coding: utf-8 -*-

"""
# File name:    utils.py
# Time :        2021/11/29 16:36
# Author:       xyguoo@163.com
# Description:  
"""
import os
import pickle
import tqdm


def merge_pickle_dir_to_dict(dir_name, target_path):
    files = os.listdir(dir_name)
    res_dir = {}
    for f_name in tqdm.tqdm(files):
        with open(os.path.join(dir_name, f_name), 'rb') as f:
            res_dir[f_name[:-4]] = pickle.load(f)
    with open(target_path, 'wb') as f:
        pickle.dump(res_dir, f)
