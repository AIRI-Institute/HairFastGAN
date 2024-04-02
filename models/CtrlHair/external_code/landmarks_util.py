# -*- coding: utf-8 -*-

"""
# File name:    landmarks_util.py
# Time :        2022/07/15
# Author:       xyguoo@163.com
# Description:  
"""
import os
import pickle as pkl

import dlib as dlib
import numpy as np
import tqdm
import cv2

detector = dlib.get_frontal_face_detector()
predictor_dict = {68: dlib.shape_predictor('models/CtrlHair/external_model_params/shape_predictor_68_face_landmarks.dat'),
                  81: dlib.shape_predictor('models/CtrlHair/external_model_params/shape_predictor_81_face_landmarks.dat')}


def detect_landmarks(root_dir, dataset_name, landmark_output_file_path, output_dir=None, predictor=None):
    result_dic = {}
    for dn in dataset_name:
        img_dir = os.path.join(root_dir, dn, 'images_256')
        files = os.listdir(img_dir)
        files.sort()

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for f in tqdm.tqdm(files):
            file_path = os.path.join(img_dir, f)
            img_rd = cv2.imread(file_path)
            img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)

            faces = detector(img_gray, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # annotate landmarks
            if len(faces) != 0:
                landmarks = np.array([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])
                result_dic['%s___%s' % (dn, f[:-4])] = landmarks / img_gray.shape[0]
                if output_dir:
                    for idx, point in enumerate(landmarks):
                        pos = (point[0], point[1])
                        cv2.circle(img_rd, pos, 2, color=(139, 0, 0))
                        cv2.putText(img_rd, str(idx + 1), pos, font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imwrite(os.path.join(output_dir, f), img_rd)
            else:
                # not detect face
                print('no face for %s' % file_path)

    with open(landmark_output_file_path, 'wb') as f:
        pkl.dump(result_dic, f)