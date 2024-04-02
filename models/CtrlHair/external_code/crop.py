# -*- coding: utf-8 -*-

"""
# File name:    crop.py
# Time :        2021/9/30 21:20
# Author:       xyguoo@163.com
# Description:  
"""

import os
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import cv2

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid "Decompressed Data Too Large" error


def recreate_aligned_images(img, lm_68, output_size=1024, transform_size=4096, enable_padding=True):
    pil_img = PIL.Image.fromarray(img)
    lm_chin = lm_68[0: 17]  # left-right
    lm_eyebrow_left = lm_68[17: 22]  # left-right
    lm_eyebrow_right = lm_68[22: 27]  # left-right
    lm_nose = lm_68[27: 31]  # top-down
    lm_nostrils = lm_68[31: 36]  # top-down
    lm_eye_left = lm_68[36: 42]  # left-clockwise
    lm_eye_right = lm_68[42: 48]  # left-clockwise
    lm_mouth_outer = lm_68[48: 60]  # left-clockwise
    lm_mouth_inner = lm_68[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    img = pil_img

    trans_points = lm_68

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
        trans_points = trans_points / shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
        trans_points = trans_points - np.array([crop[0], crop[1]])

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        trans_points = trans_points + np.array([pad[0], pad[1]])
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    trans_data = (quad + 0.5)
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, trans_data.flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    projective_matrix = cv2.getPerspectiveTransform(trans_data.astype('float32'),
                                                    np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype='float32'))
    augmented_lm = projective_matrix @ np.concatenate([trans_points, np.ones([trans_points.shape[0], 1])], axis=1).T
    trans_points = augmented_lm[:2, :] / augmented_lm[2] * output_size
    trans_points = trans_points.T
    trans_points = (trans_points + 0.5).astype('int32')
    return img, trans_points[:68]


def draw_landmarks(landmarks, img_np, font_size=1.0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.circle(img_np, pos, 2, color=(139, 0, 0))
        cv2.putText(img_np, str(idx + 1), pos, font, font_size, (0, 0, 255), 1, cv2.LINE_AA)
    return img_np

