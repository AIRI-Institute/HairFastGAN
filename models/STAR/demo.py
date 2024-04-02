import os
import cv2
import copy
import dlib
import math
import argparse
import numpy as np
import gradio as gr
from matplotlib import pyplot as plt
import torch
# private package
from lib import utility


class GetCropMatrix():
    """
    from_shape -> transform_matrix
    """

    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix


class TransformPerspective():
    """
    image, matrix3x3 -> transformed_image
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)


class TransformPoints2D():
    """
    points (nx2), matrix (3x3) -> points (nx2)
    """

    def process(self, srcPoints, matrix):
        # nx3
        desPoints = np.concatenate([srcPoints, np.ones_like(srcPoints[:, [0]])], axis=1)
        desPoints = desPoints @ np.transpose(matrix)  # nx3
        desPoints = desPoints[:, :2] / desPoints[:, [2, 2]]
        return desPoints.astype(srcPoints.dtype)


class Alignment:
    def __init__(self, args, model_path, dl_framework, device_ids):
        self.input_size = 256
        self.target_face_scale = 1.0
        self.dl_framework = dl_framework

        # model
        if self.dl_framework == "pytorch":
            # conf
            self.config = utility.get_config(args)
            self.config.device_id = device_ids[0]
            # set environment
            utility.set_environment(self.config)
            self.config.init_instance()
            if self.config.logger is not None:
                self.config.logger.info("Loaded configure file %s: %s" % (args.config_name, self.config.id))
                self.config.logger.info("\n" + "\n".join(["%s: %s" % item for item in self.config.__dict__.items()]))

            net = utility.get_net(self.config)
            if device_ids == [-1]:
                checkpoint = torch.load(model_path, map_location="cpu")
            else:
                checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint["net"])
            net = net.to(self.config.device_id)
            net.eval()
            self.alignment = net
        else:
            assert False

        self.getCropMatrix = GetCropMatrix(image_size=self.input_size, target_face_scale=self.target_face_scale,
                                           align_corners=True)
        self.transformPerspective = TransformPerspective(image_size=self.input_size)
        self.transformPoints2D = TransformPoints2D()

    def norm_points(self, points, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            return points / torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            return (points * 2 + 1) / torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1

    def denorm_points(self, points, align_corners=False):
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            return (points + 1) / 2 * torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            return ((points + 1) * torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1) / 2

    def preprocess(self, image, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)
        input_tensor = input_tensor[np.newaxis, :]

        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.float().permute(0, 3, 1, 2)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        input_tensor = input_tensor.to(self.config.device_id)
        return input_tensor, matrix

    def postprocess(self, srcPoints, coeff):
        # dstPoints = self.transformPoints2D.process(srcPoints, coeff)
        # matrix^(-1) * src = dst
        # src = matrix * dst
        dstPoints = np.zeros(srcPoints.shape, dtype=np.float32)
        for i in range(srcPoints.shape[0]):
            dstPoints[i][0] = coeff[0][0] * srcPoints[i][0] + coeff[0][1] * srcPoints[i][1] + coeff[0][2]
            dstPoints[i][1] = coeff[1][0] * srcPoints[i][0] + coeff[1][1] * srcPoints[i][1] + coeff[1][2]
        return dstPoints

    def analyze(self, image, scale, center_w, center_h):
        input_tensor, matrix = self.preprocess(image, scale, center_w, center_h)

        if self.dl_framework == "pytorch":
            with torch.no_grad():
                output = self.alignment(input_tensor)
            landmarks = output[-1][0]
        else:
            assert False

        landmarks = self.denorm_points(landmarks)
        landmarks = landmarks.data.cpu().numpy()[0]
        landmarks = self.postprocess(landmarks, np.linalg.inv(matrix))

        return landmarks


def draw_pts(img, pts, mode="pts", shift=4, color=(0, 255, 0), radius=1, thickness=1, save_path=None, dif=0,
             scale=0.3, concat=False, ):
    img_draw = copy.deepcopy(img)
    for cnt, p in enumerate(pts):
        if mode == "index":
            cv2.putText(img_draw, str(cnt), (int(float(p[0] + dif)), int(float(p[1] + dif))), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness)
        elif mode == 'pts':
            if len(img_draw.shape) > 2:
                # 此处来回切换是因为opencv的bug
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
            cv2.circle(img_draw, (int(p[0] * (1 << shift)), int(p[1] * (1 << shift))), radius << shift, color, -1,
                       cv2.LINE_AA, shift=shift)
        else:
            raise NotImplementedError
    if concat:
        img_draw = np.concatenate((img, img_draw), axis=1)
    if save_path is not None:
        cv2.imwrite(save_path, img_draw)
    return img_draw


def process(input_image):
    image_draw = copy.deepcopy(input_image)
    dets = detector(input_image, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
        exit()

    results = []
    for detection in dets:
        face = sp(input_image, detection)
        shape = []
        for i in range(68):
            x = face.part(i).x
            y = face.part(i).y
            shape.append((x, y))
        shape = np.array(shape)
        # image_draw = draw_pts(image_draw, shape)
        x1, x2 = shape[:, 0].min(), shape[:, 0].max()
        y1, y2 = shape[:, 1].min(), shape[:, 1].max()
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2

        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        landmarks_pv = alignment.analyze(input_image, scale, center_w, center_h)
        results.append(landmarks_pv)
        image_draw = draw_pts(image_draw, landmarks_pv)
    return image_draw, results


if __name__ == '__main__':
    # face detector
    # could be downloaded in this repo: https://github.com/italojs/facial-landmarks-recognition/tree/master
    predictor_path = '/path/to/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    # facial landmark detector
    args = argparse.Namespace()
    args.config_name = 'alignment'
    # could be downloaded here: https://drive.google.com/file/d/1aOx0wYEZUfBndYy_8IYszLPG_D2fhxrT/view
    model_path = '/path/to/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl'
    device_ids = '0'
    device_ids = list(map(int, device_ids.split(",")))
    alignment = Alignment(args, model_path, dl_framework="pytorch", device_ids=device_ids)

    # image:      input image
    # image_draw: draw the detected facial landmarks on image
    # results:    a list of detected facial landmarks
    face_file_path = '/path/to/face/image/bald_guys.jpg'
    image = cv2.imread(face_file_path)
    image_draw, results = process(image)

    # visualize
    img = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    # demo
    # interface = gr.Interface(fn=process, inputs="image", outputs="image")
    # interface.launch(share=True)
