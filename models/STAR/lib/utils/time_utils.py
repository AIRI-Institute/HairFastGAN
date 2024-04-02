# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time, sys
import numpy as np


def time_for_file():
    ISOTIMEFORMAT = '%d-%h-at-%H-%M-%S'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def time_string_short():
    ISOTIMEFORMAT = '%Y%m%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def time_print(string, is_print=True):
    if (is_print):
        print('{} : {}'.format(time_string(), string))


def convert_size2str(torch_size):
    dims = len(torch_size)
    string = '['
    for idim in range(dims):
        string = string + ' {}'.format(torch_size[idim])
    return string + ']'


def convert_secs2time(epoch_time, return_str=False):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    if return_str:
        str = '[Time Left: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        return str
    else:
        return need_hour, need_mins, need_secs
