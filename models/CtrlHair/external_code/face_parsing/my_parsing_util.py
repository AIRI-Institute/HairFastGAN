# -*- coding: utf-8 -*-

"""
# File name:    my_parsing_util.py
# Time :        2022/07/15
# Author:       xyguoo@163.com
# Description:  
"""
import sys

import numpy as np
import torch
from PIL import Image

from models.CtrlHair.external_code.face_parsing.model import BiSeNet
from models.CtrlHair.global_value_utils import PARSING_LABEL_LIST
import torchvision.transforms as transforms


class FaceParsing:
    label_list = {0: 'background', 1: 'skin_other', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye',
                  5: 'r_eye', 6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 9: 'ear_r',
                  10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck',
                  15: 'neck_l', 16: 'cloth', 17: 'hair', 18: 'hat'}
    skin_area = {1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13}
    bise_net = None
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    @staticmethod
    def parsing_img(img):
        pil_img = Image.fromarray(img)
        with torch.no_grad():
            image = pil_img.resize((512, 512), Image.BILINEAR)
            img = FaceParsing.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            if FaceParsing.bise_net is None:
                n_classes = 19
                FaceParsing.bise_net = BiSeNet(n_classes=n_classes)
                # FaceParsing.bise_net.cuda()
                save_pth = 'models/CtrlHair/external_model_params/face_parsing_79999_iter.pth'
                FaceParsing.bise_net.load_state_dict(torch.load(save_pth))
                FaceParsing.bise_net.eval()
            out = FaceParsing.bise_net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing, image

    @staticmethod
    def swap_parsing_label_to_celeba_mask(parsing):
        celeba_parsing = np.zeros_like(parsing)
        label_lists = list(FaceParsing.label_list.values())
        for label_idx, label_name in enumerate(PARSING_LABEL_LIST):
            celeba_parsing[label_lists.index(label_name) == parsing] = label_idx
        return celeba_parsing


class FaceParsing_tensor:
    label_list = {0: 'background', 1: 'skin_other', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye',
                  5: 'r_eye', 6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 9: 'ear_r',
                  10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck',
                  15: 'neck_l', 16: 'cloth', 17: 'hair', 18: 'hat'}
    skin_area = {1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13}
    bise_net = None
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    @staticmethod
    def parsing_img(img=None):
        with torch.no_grad():
            if FaceParsing.bise_net is None:
                n_classes = 19
                FaceParsing.bise_net = BiSeNet(n_classes=n_classes)
                save_pth = 'pretrained_models/BiSeNet/face_parsing_79999_iter.pth'
                FaceParsing.bise_net.load_state_dict(torch.load(save_pth))
                FaceParsing.bise_net.eval()
                FaceParsing.bise_net = FaceParsing.bise_net.cuda()

            if img is None:
                return
            out = FaceParsing.bise_net(img)[0]
            parsing = out.squeeze(0).argmax(0)
        return parsing, None

    @staticmethod
    def swap_parsing_label_to_celeba_mask(parsing):
        celeba_parsing = torch.zeros_like(parsing)
        label_lists = list(FaceParsing.label_list.values())
        for label_idx, label_name in enumerate(PARSING_LABEL_LIST):
            celeba_parsing[label_lists.index(label_name) == parsing] = label_idx
        return celeba_parsing
