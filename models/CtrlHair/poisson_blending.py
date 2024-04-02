# -*- coding: utf-8 -*-

"""
# File name:    poisson_blending.py
# Time :        2021/11/5 15:22
# Author:       xyguoo@163.com
# Description:  
"""

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    return mat_A


def poisson_blending(source, target, mask, with_gamma=True):
    """
    source: H * W * 3, cv2 image
    target: H * W * 3, cv2 image
    mask:   H * W * 1
    """
    if with_gamma:
        gamma_value = 2.2
    else:
        gamma_value = 1
    source = source.astype('float')
    target = target.astype('float')
    source = np.power(source, 1 / gamma_value)
    target = np.power(target, 1 / gamma_value)

    res = target.copy()
    y_range, x_range = source.shape[:2]
    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()
    mask[mask != 0] = 1

    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()

    y_min, y_max = 0, y_range
    x_min, x_max = 0, x_range

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat) * alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        res[:, :, channel] = x

    res = np.power(res, gamma_value)

    res[res > 255] = 255
    res[res < 0] = 0
    res = res.astype('uint8')
    return res
