# -*- coding: utf-8 -*-

"""
# File name:    script_adaptor_test_pool.py
# Time :        2022/07/14
# Author:       xyguoo@163.com
# Description:  
"""

import sys
sys.path.append('.')

from shape_branch.adaptor_generation import AdaptorPoolGeneration
from shape_branch.config import cfg

if __name__ == '__main__':
    pp = AdaptorPoolGeneration(only_hair=False,
                               dir_name=cfg.adaptor_pool_dir,
                               test_dir_name=cfg.adaptor_test_pool_dir, thread_num=10)

    ######################################################
    # Run this for generating wrap pool for testing
    ######################################################
    pp.generate_test_set(cfg.sample_batch_size)
