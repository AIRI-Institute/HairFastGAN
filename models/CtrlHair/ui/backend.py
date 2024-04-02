# -*- coding: utf-8 -*-

"""
# File name:    hair_swap.py
# Time :        2022/2/20 17:06
# Author:       xyguoo@163.com
# Description:  This is the backend interface for editing. When you want to customize custom editing and
                modification functions (such as replacing mask), it is recommended to call these interfaces directly.
                In the finally main section of this file, there is an example of using Backend
"""

import sys
sys.path.append('.')

import copy
import os

import cv2
import torch

from global_value_utils import HAIR_IDX, TEMP_FOLDER
from my_torchlib.train_utils import generate_noise
from shape_branch.shape_util import mask_label_to_one_hot, mask_one_hot_to_label, split_hair_face
from util.mask_color_util import mask_to_rgb
from wrap_codes.mask_adaptor import wrap_by_imgs

from hair_editor import HairEditor
from util.color_from_hsv_to_gaussian import DistTranslation


class LatentRepresentation:
    def __init__(self):
        self.color = None
        self.curliness = None
        self.shape = None
        self.texture = None
        self.face = None


class Backend(HairEditor):
    """
    This is the main interface set, please call these interfaces directly to customize custom editing
    """

    def __init__(self, maximum_value_fe, blending=True, temp_path=os.path.join(TEMP_FOLDER, 'demo_output')):
        """
        :param maximum_value_fe: The max value in sliding bar of the frontend ui. If you just use the backend,
            but not use the frontend, this value is
        :param blending: Whether use poisson blending between hair region and others region
        :param temp_path: the dir that save temp content, which can be removed after execution
        """
        super().__init__(True, True)
        self.target_img = None
        self.input_img = None
        self.target_mask = None
        self.input_mask = None
        self.cur_latent = None
        self.target_latent = None
        self.cur_mask = None
        self.input_sean_code = None
        self.target_size = 256
        self.maximum_value_fe = maximum_value_fe
        self.temp_path = temp_path
        self.blending = blending
        self.dist_translation = DistTranslation()

    def parse_img(self, img_rgb, target_img=False):
        """
        mask the input image
        :param img_rgb: with channel order rgb
        :param target_img: is the target image?
        """
        img_ts = cv2.resize(img_rgb, (self.target_size, self.target_size))
        mask = self.get_mask(img_rgb)
        lr = LatentRepresentation()

        # infer shape
        if target_img:
            out_mask = None
        else:
            mask_batch = self.preprocess_mask(mask)
            mask_tensor = torch.tensor(mask_batch, dtype=torch.uint8, device=self.device)
            mask_one_hot = mask_label_to_one_hot(mask_tensor)
            hair, face = split_hair_face(mask_one_hot)
            hair_code = self.mask_generator.forward_hair_encoder(hair, testing=True)
            face_code = self.mask_generator.forward_face_encoder(face)
            lr.shape = hair_code
            lr.face = face_code
            out_mask = self.mask_generator.forward_decode_by_code(hair_code, face_code)
            out_mask = mask_one_hot_to_label(out_mask).cpu().numpy()[0]

        # infer feature
        input_code = self.get_code(self.preprocess_img(img_rgb), mask_batch)
        hair_feature = input_code[:, HAIR_IDX]

        out_color = self.feature_rgb_predictor({'code': hair_feature})

        c = out_color['rgb_mean'].detach().cpu().numpy()
        c_hsv = cv2.cvtColor(c[None, ...].astype('uint8'), cv2.COLOR_RGB2HSV)
        c_hsv = torch.tensor(c_hsv).to(self.device)[0]
        lr.color = {'hsv': c_hsv, 'pca_std': out_color['pca_std']}

        out_enc = self.feature_encoder({'code': hair_feature})
        lr.curliness = out_enc['noise_curliness']
        lr.texture = out_enc['noise']
        return img_ts, out_mask, lr, mask, input_code, hair_feature

    def tensor_hsv_to_rgb(self, hsv):
        """
        input tensor with hsv and convert to rgb
        """
        c = hsv.detach().cpu().numpy()
        c_rgb = cv2.cvtColor(c[None, ...].astype('uint8'), cv2.COLOR_HSV2RGB)
        c_rgb = torch.tensor(c_rgb).to(self.device)[0]
        return c_rgb

    def tensor_rgb_to_hsv(self, rgb):
        """
        input tensor with rgb and convert to hsv
        """

        c = rgb.detach().cpu().numpy()
        c_hsv = cv2.cvtColor(c[None, ...].astype('uint8'), cv2.COLOR_RGB2HSV)
        c_hsv = torch.tensor(c_hsv).to(self.device)[0]
        return c_hsv

    def set_input_img(self, img_rgb):
        """
        set and parse the input image
        :param img_rgb:
        """
        self.input_img, self.cur_mask, self.cur_latent, \
        self.input_mask, self.input_sean_code, self.input_hair_feature = self.parse_img(img_rgb)
        input_mask_show = mask_to_rgb(self.cur_mask, draw_type=1)
        return self.input_img, input_mask_show

    def set_target_img(self, img_rgb):
        """
        set and parse the target image
        :param img_rgb:
        """
        self.target_img, _, self.target_latent, \
        self.target_mask, _, self.target_hair_feature = self.parse_img(img_rgb)
        input_maks_show = mask_to_rgb(self.target_mask, draw_type=1)
        return self.target_img, input_maks_show

    def output(self, target_latent=None, feature=None):
        """
        generate an color_texture image
        :param target_latent: if the latent representation of target is not provided,
                `self.cur_latent` and `self.cur_mask` will be used
        :param feature: if edited hair feature X is not provided, it will be generated with color and texture branch
        :return: color_texture image
        """
        if target_latent is None:
            target_latent = self.cur_latent
            target_mask = self.cur_mask
        else:
            target_mask = self.refresh_cur_mask(target_latent)[0]

        if 'rgb_mean' in target_latent.color:
            target_color_rgb = self.target_latent.color['rgb_mean']
        else:
            target_color_rgb = self.tensor_hsv_to_rgb(target_latent.color['hsv'])

        if feature is None:
            data = {'noise': target_latent.texture, 'noise_curliness': target_latent.curliness,
                    'rgb_mean': target_color_rgb, 'pca_std': target_latent.color['pca_std']}
            feature = self.feature_generator(data)['code']
        self.input_sean_code[:, HAIR_IDX] = feature
        edit_img = self.gen_img(self.input_sean_code, target_mask[None, None, ...])
        output_img, _ = self.postprocess_blending(self.input_img, edit_img, self.input_mask, target_mask,
                                                  blending=self.blending)

        return output_img

    def change_curliness(self, val):
        """
        change the latent representation of curliness of texture
        """
        self.cur_latent.curliness[0] = val

    # def change_color(self, val, idx):
    #     val = (val + self.maximum_value_fe) / 2 / self.maximum_value_fe
    #     if idx == 3:
    #         self.cur_latent.color['pca_std'][0] = val * 100 + 20
    #     else:
    #         if idx == 0:
    #             val *= 179
    #         if idx == 1:
    #             val *= 255
    #         if idx == 2:
    #             val *= 255
    #         self.cur_latent.color['hsv'][0][idx] = val

    def change_color(self, val, idx):
        """
        change the latent representation of color
        idx=0 is hue, idx=1 is saturation, idx=2 is brightness, idx=3 is variance

        please note that when idx=3, the variance is not obey gaussian,
        the value range is in [-self.maximum_value_fe, self.maximum_value_fe]
        """
        if idx == 3:
            val = (val + self.maximum_value_fe) / 2 / self.maximum_value_fe
            self.cur_latent.color['pca_std'][0] = val * 100 + 20
        else:
            val = self.dist_translation.gaussian_to_val(idx, val)
            self.cur_latent.color['hsv'][0][idx] = val

    def change_shape(self, val, idx):
        """
        change the latent representation of shape
        :param val: latent value
        :param idx: for current checkpoint, idx=0 is length, idx=1 is volumn, idx=2 is bangs_direction, idx=3 is bangs
        """
        self.continue_change_with_direction('shape', self.shape_dirs[idx], val)
        self.refresh_cur_mask()

    def change_texture(self, val, idx):
        """
        change the latent representation of texture
        :param val: latent value
        :param idx: for current checkpoint, idx=0 is smoothness, idx=1 is thickness
        """
        self.continue_change_with_direction('texture', self.texture_dirs[idx], val)

    def get_curliness_be2fe(self):
        """
        convert the latent representation to the value of sliding bar in the frontend of Pyqt ui, and return it
        """
        return self.cur_latent.curliness[0]

    def get_color_be2fe(self):
        """
        convert the latent representation to the value of sliding bar in the frontend of Pyqt ui, and return it
        """
        c_hsv = self.cur_latent.color['hsv'].detach().cpu().numpy()[0]
        # color0 = c_hsv[0] / 179 * 2 * self.maximum_value_fe - self.maximum_value_fe
        # color1 = c_hsv[1] / 255 * 2 * self.maximum_value_fe - self.maximum_value_fe
        # color2 = c_hsv[2] / 255 * 2 * self.maximum_value_fe - self.maximum_value_fe
        color0 = self.dist_translation.val_to_gaussian(0, c_hsv[0])
        color1 = self.dist_translation.val_to_gaussian(1, c_hsv[1])
        color2 = self.dist_translation.val_to_gaussian(2, c_hsv[2])
        var_fe = (self.cur_latent.color['pca_std'][0] - 20) / 100 * 2 * self.maximum_value_fe - self.maximum_value_fe
        return color0, color1, color2, var_fe

    def get_shape_be2fe(self):
        """
        convert the latent representation to the value of sliding bar in the frontend of Pyqt ui, and return it
        """
        res = []
        for idx in range(4):
            res.append(torch.dot(self.cur_latent.shape[0], self.shape_dirs[idx]))
        return res

    def get_texture_be2fe(self):
        """
        convert the latent representation to the value of sliding bar in the frontend of Pyqt ui, and return it
        """
        res = []
        for idx in range(2):
            res.append(torch.dot(self.cur_latent.texture[0], self.texture_dirs[idx]))
        return res

    def transfer_latent_representation(self, flag, refresh=True):
        """
        transfer the latent representation of target image to input image.
        i.e., transfer from self.target_latent to self.cur_latent
        :param flag: 'color', 'texture' or 'shape'
        :param refresh: whether refresh mask
        :return:
        """
        if flag == 'shape':
            wt, _ = wrap_by_imgs(self.target_img, self.input_img, wrap_temp_folder=self.temp_path,
                                 need_crop=False)
            wt = self.preprocess_mask(wt)
            self.warp_target = wt[0, 0]
            mask_tensor = torch.tensor(wt, dtype=torch.uint8, device=self.device)
            mask_one_hot = mask_label_to_one_hot(mask_tensor)
            hair, face = split_hair_face(mask_one_hot)
            hair_code = self.mask_generator.forward_hair_encoder(hair, testing=True)
            face_code = self.mask_generator.forward_face_encoder(face)
            self.target_latent.shape = hair_code
            self.target_latent.face = face_code

            self.refresh_cur_mask()

        target_att = self.target_latent.__getattribute__(flag)
        if isinstance(target_att, torch.Tensor):
            self.cur_latent.__setattr__(flag, target_att.clone())
        else:
            cp_dict = copy.copy(target_att)
            for ke in cp_dict:
                cp_dict[ke] = cp_dict[ke].clone()
            self.cur_latent.__setattr__(flag, cp_dict)

        if flag == 'shape' and refresh:
            self.refresh_cur_mask()

        if flag == 'texture':
            self.transfer_latent_representation('curliness')

    def refresh_cur_mask(self, target_latent=None):
        """
        refresh and generate current mask
        :param target_latent: if no target latent is given, self.cur_mask will be used
        :return:
        """
        if target_latent is None:
            target_latent = self.cur_latent
        out_mask = self.mask_generator.forward_decode_by_code(target_latent.shape, target_latent.face)
        out_mask = mask_one_hot_to_label(out_mask).cpu().numpy()[0]
        self.cur_mask = out_mask
        return out_mask, mask_to_rgb(out_mask, draw_type=1)

    def get_cur_mask(self):
        """
        get the mask, which can be visited
        """
        return mask_to_rgb(self.cur_mask, draw_type=1)

    def interpolate_hsv(self, hsv1, hsv2, alpha):
        """
        final hsv = hsv1 * (1 - alpha) + hsv2 * alpha
        return: final hsv
        """
        rgb1 = self.tensor_hsv_to_rgb(hsv1)
        rgb2 = self.tensor_hsv_to_rgb(hsv2)
        rgb = rgb1 * (1 - alpha) + rgb2 * alpha
        inter_hsv = self.tensor_rgb_to_hsv(rgb)
        return inter_hsv

    def interpolate_triple(self, latent1, latent2, latent3, alpha1, alpha2, alpha3):
        """
        (latent1 * alpha1 + latent2 * alpha2 + latent3 * alpha3) / (alpha1 + alpha2 + alpha3)
        :return: final latent
        """
        latent12 = self.interpolate(latent1, latent2, alpha2 / (alpha1 + alpha2))
        latent_res = self.interpolate(latent12, latent3, alpha3)
        return latent_res

    def interpolate(self, latent1, latent2, alpha):
        """
        final latent = latent1 * (1 - alpha) + latent2 * alpha
        :return: final latent
        """
        result_latent = LatentRepresentation()
        for att in ['curliness', 'shape', 'texture']:
            result_latent.__setattr__(att, latent1.__getattribute__(att) * (1 - alpha) +
                                      latent2.__getattribute__(att) * alpha)
        color_dic = {}
        color_dic['pca_std'] = latent1.color['pca_std'] * (1 - alpha) + latent2.color['pca_std'] * alpha
        color_dic['hsv'] = self.interpolate_hsv(latent1.color['hsv'], latent2.color['hsv'], alpha)

        result_latent.color = color_dic
        result_latent.face = self.cur_latent.face
        return result_latent

    def interpolate_each_att(self, latent1, latent2, alpha, att_name):
        """
        interpolate a certain latent
        :param att_name: curliness, shape, texture, color
        :return: final full latent
        """
        result_latent = LatentRepresentation()
        for att in ['curliness', 'shape', 'texture']:
            result_latent.__setattr__(att, self.cur_latent.__getattribute__(att).clone())

        if att_name == 'shape':
            # keep color
            color_dic = {}
            for semantic in ['hsv', 'pca_std']:
                color_dic[semantic] = self.cur_latent.color[semantic].clone()
            result_latent.__setattr__(att_name, latent1.__getattribute__(att_name) * (1 - alpha) +
                                      latent2.__getattribute__(att_name) * alpha)
        elif att_name in ['curliness', 'texture']:
            # keep color
            color_dic = {}
            for semantic in ['hsv', 'pca_std']:
                color_dic[semantic] = self.cur_latent.color[semantic].clone()
            result_latent.__setattr__('curliness', latent1.__getattribute__('curliness') * (1 - alpha) +
                                      latent2.__getattribute__('curliness') * alpha)
            result_latent.__setattr__('texture', latent1.__getattribute__('texture') * (1 - alpha) +
                                      latent2.__getattribute__('texture') * alpha)
        else:
            color_dic = {}
            # for semantic in ['hsv', 'pca_std']:
            color_dic['pca_std'] = latent1.color['pca_std'] * (1 - alpha) + latent2.color['pca_std'] * alpha
            color_dic['hsv'] = self.interpolate_hsv(latent1.color['hsv'], latent2.color['hsv'], alpha)

        result_latent.color = color_dic
        result_latent.face = self.cur_latent.face

        return result_latent

    @staticmethod
    def show_hair_region(mask, non_hair_value=0):
        """
        Get hair region, and full none hair  region as `non_hair_value`
        :param mask:
        :param non_hair_value:
        :return:
        """
        mask_rgb = mask_to_rgb(mask, draw_type=1)
        mask_rgb[mask != HAIR_IDX] = non_hair_value
        return mask_rgb

    def directly_change_hair_mask(self, hair_mask):
        """
        Directly replace hair region with a hair_mask. This is a recommend method to imply mask transfer.
        :param hair_mask:
        :return:
        """
        hair_mask = hair_mask == HAIR_IDX
        face_logit = self.mask_generator.forward_face_decoder(self.cur_latent.face)
        hair_logit = torch.tensor(hair_mask)[None, None, ...].type_as(face_logit).to(self.device)
        hair_logit = hair_logit * (face_logit.max() - face_logit.min() + 2) + face_logit.min() - 1
        mask = self.mask_generator.forward_decoder(hair_logit, face_logit)
        self.cur_mask = mask_one_hot_to_label(mask).cpu().numpy()[0]

    def get_random_texture(self):
        """
        sample a texture latent code randomly
        :return:
        """
        random_latent = generate_noise(1, 8)
        random_latent = random_latent.to(self.device)
        self.cur_latent.texture = random_latent

    def get_random_shape(self):
        """
        sample a shape latent code randomly
        :return:
        """
        random_latent = generate_noise(1, 16)
        random_latent = random_latent.to(self.device)
        self.cur_latent.shape = random_latent
        self.refresh_cur_mask()

    def get_random_curliness(self):
        """
        sample a curliness latent code randomly
        :return:
        """
        random_latent = generate_noise(1, 1)
        random_latent = random_latent.to(self.device)
        self.cur_latent.curliness = random_latent

    def continue_change_with_direction(self, att_name, direction, val):
        """
        change the latent code value on a projection direction
        :param att_name: shape or texture
        :param direction: projection direction
        :param val: coordinate on this projection direction
        :return:
        """
        att = self.cur_latent.__getattribute__(att_name)
        att = att + (val - torch.dot(att[0], direction)) * direction
        self.cur_latent.__setattr__(att_name, att)
        if att_name == 'shape':
            self.refresh_cur_mask()


"""
This is a example of using Backend for costume editing
"""
if __name__ == '__main__':
    be = Backend(2.5)
    from util.imutil import read_rgb, write_rgb

    input_image = read_rgb('imgs/00079.png')
    target_image = read_rgb('imgs/00001.png')

    """
    If the image need crop
    """
    # input_image = be.crop_face(input_image)
    # target_image = be.crop_face(target_image)

    input_image = cv2.resize(input_image, (256, 256))

    be.set_input_img(input_image)
    be.set_target_img(target_image)

    # transfer all latent code from target image to input image
    be.transfer_latent_representation('texture')
    be.transfer_latent_representation('color')
    be.transfer_latent_representation('shape')

    # change the variance manually
    be.change_color(1.0, 2)

    out_mask = be.get_mask(input_image)
    output_img = be.output()
    write_rgb('temp.png', output_img)
    # above is the output image

    im2 = read_rgb('imgs/00037.png')
    im2 = cv2.resize(im2, (256, 256))
    be.set_target_img(im2)
    be.transfer_latent_representation('shape')
    output_img2 = be.output()
    # above is the output image 2
