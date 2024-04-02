# -*- coding: utf-8 -*-

"""
# File name:    find_semantic_direction.py
# Time :        2022/07/17
# Author:       xyguoo@163.com
# Description:  
"""
import torch


def get_random_direction(dim, existing_dirs):
    dir = torch.randn(dim)

    for dd in existing_dirs:
        dir = dir - torch.dot(dir, dd) * dd

    if dir[0] < 0:
        dir = -dir
    dir = dir / dir.norm()
    return dir
