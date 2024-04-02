# -*- coding: utf-8 -*-

"""
# File name:    hair_editor.py
# Time :        2021/11/18 17:21
# Author:       xyguoo@163.com
# Description:  
"""

import os
import pickle
from glob import glob

import cv2
import numpy as np
import torch

import my_torchlib
from color_texture_branch.solver import Solver as SolveFeature
from external_code.face_parsing.my_parsing_util import FaceParsing
from global_value_utils import HAIR_IDX, PARSING_LABEL_LIST
from poisson_blending import poisson_blending
from sean_codes.models.pix2pix_model import Pix2PixModel
from sean_codes.options.test_options import TestOptions
from shape_branch.solver import Solver as SolverMask
from util.imutil import write_rgb


# adaptor_root_dir = '/data1/guoxuyang/myWorkSpace/hair_editing'
# sys.path.append(adaptor_root_dir)
# sys.path.append(os.path.join(adaptor_root_dir, 'external_code/face_3DDFA'))


def change_status(model, new_status):
    for m in model.modules():
        if hasattr(m, 'status'):
            m.status = new_status


class HairEditor:
    """
    This is the basic module, that could achieve many editing task. ui/hair_swap.py/Backend succeed this class.
    """

    def __init__(self, load_feature_model=True, load_mask_model=True):
        self.opt = TestOptions().parse()
        self.opt.status = 'test'
        self.sean_model = Pix2PixModel(self.opt)
        self.sean_model.eval()
        self.img_size = 256
        self.device = torch.device('cuda', 0)

        if load_feature_model:
            from color_texture_branch.config import cfg as cfg_feature
            self.solver_feature = SolveFeature(cfg_feature, device=self.device, local_rank=-1, training=False)

            self.feature_encoder = self.solver_feature.dis
            self.feature_generator = self.solver_feature.gen
            self.feature_rgb_predictor = self.solver_feature.rgb_model
            # self.feature_curliness_predictor = self.solver_feature.curliness_model

            # ckpt_dir = 'external_model_params/disentangle_checkpoints/' + cfg_app.experiment_name + '/checkpoints'
            ckpt_dir = 'model_trained/color_texture/' + cfg_feature.experiment_name + '/checkpoints'
            ckpt = my_torchlib.load_checkpoint(ckpt_dir)
            for model_name in ['Model_G', 'Model_D']:
                cur_model = ckpt[model_name]
                if list(cur_model)[0].startswith('module'):
                    ckpt[model_name] = {kk[7:]: cur_model[kk] for kk in cur_model}

            self.feature_generator.load_state_dict(ckpt['Model_G'], strict=True)
            self.feature_encoder.load_state_dict(ckpt['Model_D'], strict=True)

            # if 'curliness' in cfg_feature.predictor:
            #     ckpt = my_torchlib.load_checkpoint(cfg_feature.predictor.curliness.root_dir + '/checkpoints')
            #     self.feature_curliness_predictor.load_state_dict(ckpt['Predictor'], strict=True)

            if 'rgb' in cfg_feature.predictor:
                ckpt = my_torchlib.load_checkpoint(cfg_feature.predictor.rgb.root_dir + '/checkpoints')
                self.feature_rgb_predictor.load_state_dict(ckpt['Predictor'], strict=True)

            # load unsupervised direction
            existing_dirs_dir = os.path.join('model_trained/color_texture', cfg_feature.experiment_name,
                                             'texture_dir_used')
            if os.path.exists(existing_dirs_dir):
                existing_dirs_list = os.listdir(existing_dirs_dir)
                existing_dirs_list.sort()
                existing_dirs = []
                for dd in existing_dirs_list:
                    with open(os.path.join(existing_dirs_dir, dd), 'rb') as f:
                        existing_dirs.append(pickle.load(f).to(self.device))
                self.texture_dirs = existing_dirs

        if load_mask_model:
            from shape_branch.config import cfg as cfg_mask
            self.solver_mask = SolverMask(cfg_mask, device=self.device, local_rank=-1, training=False)
            self.mask_generator = self.solver_mask.gen

            ##############################################
            #  change to your checkpoints dir            #
            ##############################################
            ckpt_dir = 'model_trained/shape/' + cfg_mask.experiment_name + '/checkpoints'
            ckpt = my_torchlib.load_checkpoint(ckpt_dir)
            for model_name in ['Model_G', 'Model_D']:
                cur_model = ckpt[model_name]
                if list(cur_model)[0].startswith('module'):
                    ckpt[model_name] = {kk[7:]: cur_model[kk] for kk in cur_model}

            self.mask_generator.load_state_dict(ckpt['Model_G'], strict=True)

            # load unsupervised direction
            existing_dirs_dir = os.path.join('model_trained/shape', cfg_mask.experiment_name, 'shape_dir_used')
            if os.path.exists(existing_dirs_dir):
                existing_dirs_list = os.listdir(existing_dirs_dir)
                existing_dirs_list.sort()
                existing_dirs = []
                for dd in existing_dirs_list:
                    with open(os.path.join(existing_dirs_dir, dd), 'rb') as f:
                        existing_dirs.append(pickle.load(f).to(self.device))
                self.shape_dirs = existing_dirs

    def preprocess_img(self, img):
        img = cv2.resize(img.astype('uint8'), (self.img_size, self.img_size))
        return (np.transpose(img, [2, 0, 1]) / 127.5 - 1.0)[None, ...]

    def preprocess_mask(self, mask_img):
        mask_img = cv2.resize(mask_img.astype('uint8'), (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)
        return mask_img[None, None, :, :]

    @staticmethod
    def load_average_feature():
        ############### load average features
        # average_style_code_folder = 'styles_test/mean_style_code/mean/'
        average_style_code_folder = 'sean_codes/styles_test/mean_style_code/median/'
        input_style_dic = {}

        ############### hard coding for categories
        for i in range(19):
            input_style_dic[str(i)] = {}
            average_category_folder_list = glob(os.path.join(average_style_code_folder, str(i), '*.npy'))
            average_category_list = [os.path.splitext(os.path.basename(name))[0] for name in
                                     average_category_folder_list]

            for style_code_path in average_category_list:
                input_style_dic[str(i)][style_code_path] = torch.from_numpy(
                    np.load(os.path.join(average_style_code_folder, str(i), style_code_path + '.npy'))).cuda()
        return input_style_dic

    def get_code(self, hair_img, hair_parsing):
        # generate style code
        data = {'label': torch.tensor(hair_parsing, dtype=torch.float32),
                'instance': torch.tensor(0),
                'image': torch.tensor(hair_img, dtype=torch.float32),
                'path': ['temp/temp_npy']}
        change_status(self.sean_model, 'test')
        hair_img_code = self.sean_model(data, mode='style_code')
        return hair_img_code

    def gen_img(self, code, parsing):
        # load style code
        if not isinstance(code, torch.Tensor):
            code = torch.tensor(code)
        obj_dic = self.load_average_feature()

        for idx in range(19):
            cur_code = code[0, idx]
            if not torch.all(cur_code == 0):
                obj_dic[str(idx)]['ACE'] = cur_code

        temp_face_image = torch.zeros((0, 3, self.img_size, self.img_size))  # place holder

        data = {'label': torch.tensor(parsing, dtype=torch.float32),
                'instance': torch.tensor(0),
                'image': torch.tensor(temp_face_image, dtype=torch.float32),
                'obj_dic': obj_dic}
        change_status(self.sean_model, 'UI_mode')
        # self.model = self.model.to(code.device)
        generated = self.sean_model(data, mode='UI_mode')[0]
        return generated

    def generate_by_sean(self, face_img_code, hair_code, target_seg):
        """
        :param face_img_code: please input with the shape [19, 512]
        :param hair_code: please input with the shape [512]
        :param target_seg:
        :return:
        """
        # load style code
        obj_dic = self.load_average_feature()

        for idx in range(19):
            if idx == HAIR_IDX:
                cur_code = hair_code
                # cur_code = face_img_code[0, idx]
            else:
                cur_code = face_img_code[idx]
            if not torch.all(face_img_code == 0):
                obj_dic[str(idx)]['ACE'] = cur_code

        data = {'label': torch.tensor(target_seg, dtype=torch.float32),
                'instance': torch.tensor(0),
                'obj_dic': obj_dic,
                'image': None}
        change_status(self.sean_model, 'UI_mode')
        generated = self.sean_model(data, mode='UI_mode')[0]
        return generated

    def generate_instance_transfer_img(self, face_img, face_parsing, hair_img, hair_parsing, target_seg, edit_data=None,
                                       temp_path='temp'):
        # generate style code
        data = {'label': torch.tensor(face_parsing, dtype=torch.float32),
                'instance': torch.tensor(0),
                'image': torch.tensor(face_img, dtype=torch.float32),
                'path': ['temp/temp_npy']}
        face_img_code = self.sean_model(data, mode='style_code')

        if hair_img is None:
            hair_img_code = face_img_code
        else:
            data = {'label': torch.tensor(hair_parsing, dtype=torch.float32),
                    'instance': torch.tensor(0),
                    'image': torch.tensor(hair_img, dtype=torch.float32),
                    'path': ['temp/temp_npy']}
            change_status(self.sean_model, 'test')
            hair_img_code = self.sean_model(data, mode='style_code')
        hair_code = hair_img_code[0, HAIR_IDX]

        if edit_data is not None:
            hair_code = self.solver_feature.edit_infer(hair_code[None, ...], edit_data)[0]

        return self.generate_by_sean(face_img_code[0], hair_code, target_seg)

    def get_hair_color(self, img):
        parsing, _ = FaceParsing.parsing_img(img)
        parsing = FaceParsing.swap_parsing_label_to_celeba_mask(parsing)
        parsing = cv2.resize(parsing.astype('uint8'), (1024, 1024), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img.astype('uint8'), (1024, 1024))
        hair_mask = (parsing == HAIR_IDX).astype('uint8')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(19, 19))
        hair_mask = cv2.erode(hair_mask, kernel, iterations=1)
        points = img[hair_mask.astype('bool')]
        moment1 = points.mean(axis=0)
        return moment1

    @staticmethod
    def draw_landmarks(img, lms):
        lms = lms / 2
        lms = lms.astype('int')
        for idx, point in enumerate(lms):
            font = cv2.FONT_HERSHEY_SIMPLEX
            pos = (point[0], point[1])
            cv2.circle(img, pos, 2, color=(139, 0, 0))
            cv2.putText(img, str(idx + 1), pos, font, 0.18, (255, 0, 0), 1, cv2.LINE_AA)
        return img

    def postprocess_blending(self, face_img, res_img, face_parsing, target_parsing, verbose_print=False, blending=True):
        """
        Blend original face img and result image with poisson blending.
        If not blend, the result image will look slightly different from original image in some details in
        non-hair region, but the image quality will be better.
        :param face_img:
        :param res_img:
        :param face_parsing:
        :param target_parsing:
        :param verbose_print:
        :param blending: If `False`, the result image will do some trivial thing like transferring data type
        :return:
        """
        if verbose_print:
            print("Post process for the result image...")

        def from_tensor_order_to_cv2(tensor_img, is_mask=False):
            if isinstance(tensor_img, torch.Tensor):
                tensor_img = tensor_img.detach().cpu().numpy()
            if len(tensor_img.shape) == 4:
                tensor_img = tensor_img[0]
            if len(tensor_img.shape) == 2:
                tensor_img = tensor_img[None, ...]
            if tensor_img.shape[2] <= 3:
                return tensor_img
            res = np.transpose(tensor_img, [1, 2, 0])
            if not is_mask:
                res = res * 127.5 + 127.5
            return res

        res_img = from_tensor_order_to_cv2(res_img)
        res_img = res_img.astype('uint8')
        if blending:

            target_parsing = from_tensor_order_to_cv2(target_parsing, is_mask=True)
            face_img = from_tensor_order_to_cv2(face_img)
            face_img = face_img.astype('uint8')

            face_parsing = from_tensor_order_to_cv2(face_parsing, is_mask=True)

            res_mask = np.logical_or(target_parsing == HAIR_IDX, face_parsing == HAIR_IDX).astype('uint8')
            kernel13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(13, 13))
            kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
            res_mask_dilated = cv2.dilate(res_mask, kernel13, iterations=1)[..., None]

            res_mask_dilated5 = cv2.dilate(res_mask, kernel5, iterations=1)[..., None]

            bg_mask = (target_parsing == PARSING_LABEL_LIST.index('background'))
            res_mask_dilated = res_mask_dilated * (1 - bg_mask) + res_mask_dilated5 * bg_mask

            face_to_hair = poisson_blending(face_img, res_img, 1 - res_mask_dilated, with_gamma=True)
            return face_to_hair, res_mask_dilated
        else:
            return res_img, None

    def crop_face(self, img_rgb, save_path=None):
        """
        crop the face part in the image to adapt the editing system
        :param img_rgb:
        :param save_path:
        :return:
        """
        from external_code.crop import recreate_aligned_images
        from external_code.landmarks_util import predictor_dict, detector

        predictor_68 = predictor_dict[68]
        bbox = detector(img_rgb, 0)[0]
        lm_68 = np.array([[p.x, p.y] for p in predictor_68(img_rgb, bbox).parts()])
        crop_img_pil, lm_68 = recreate_aligned_images(img_rgb, lm_68, output_size=self.img_size)
        img_rgb = np.array(crop_img_pil)
        if save_path is not None:
            write_rgb(save_path, img_rgb)
        return img_rgb

    def get_mask(self, img_rgb):
        parsing, _ = FaceParsing.parsing_img(img_rgb)
        parsing = FaceParsing.swap_parsing_label_to_celeba_mask(parsing)
        mask_img = cv2.resize(parsing.astype('uint8'), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return mask_img
