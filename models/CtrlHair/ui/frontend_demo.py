# -*- coding: utf-8 -*-

"""
# File name:    frontend.py
# Time :        2022/2/20 15:58
# Author:       xyguoo@163.com
# Description:  This is the demo frontend
"""

import sys
sys.path.append('.')

from global_value_utils import TEMP_FOLDER
import argparse
import os

from util.common_options import ctrl_hair_parser_options

parser = argparse.ArgumentParser()
ctrl_hair_parser_options(parser)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from ui.backend import Backend
from util.imutil import read_rgb, write_rgb

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QLabel, QGridLayout, \
    QSlider, QFileDialog
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.temp_path = os.path.join(TEMP_FOLDER, 'demo_output')
        self.maximum_value = 2.0
        self.blending = not args.no_blending
        self.backend = Backend(self.maximum_value, blending=self.blending)
        self.initUI()
        self.target_size = 256
        self.need_crop = args.need_crop
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        self.font = QFont()
        self.font.setPointSize(15)
        self.setFont(self.font)

        self.input_name = None
        self.target_name = None

    def initUI(self):
        self.lbl_target_img = QLabel(self)
        self.lbl_input_img = QLabel(self)
        self.lbl_input_seg = QLabel(self)
        self.lbl_out_img = QLabel(self)

        self.labels = [self.lbl_target_img, self.lbl_input_img,
                       self.lbl_input_seg, self.lbl_out_img]

        self.grid1 = QGridLayout()
        # tags = ['target image', 'input image', 'hair shape', 'color_texture']
        # for idx in range(len(self.labels)):
        #     self.grid1.addWidget(QLabel(tags[idx]), 0, idx)
        for idx in range(len(self.labels)):
            self.grid1.addWidget(self.labels[idx], 1, idx, alignment=Qt.AlignTop)
            self.labels[idx].setFixedSize(256, 256)

        self.btn_open_target = QPushButton('Target Image', self)
        self.btn_open_target.clicked[bool].connect(self.evt_open_target)
        self.grid1.addWidget(self.btn_open_target, 0, 0)

        self.btn_open_input = QPushButton('Input Image', self)
        self.btn_open_input.clicked[bool].connect(self.evt_open_input)
        self.grid1.addWidget(self.btn_open_input, 0, 1)

        self.grid1.addWidget(QLabel('Hair Shape'), 0, 2, alignment=Qt.AlignCenter)

        self.btn_output = QPushButton('Output', self)
        self.btn_output.clicked[bool].connect(self.evt_output)
        self.grid1.addWidget(self.btn_output, 0, 3)
        self.btn_output.setEnabled(False)

        self.grid2 = QGridLayout()

        self.btn_trans_color = QPushButton('Transfer Color', self)
        self.btn_trans_color.clicked[bool].connect(self.evt_trans_color)
        self.grid2.addWidget(self.btn_trans_color, 10, 0)
        self.btn_trans_color.setEnabled(False)

        self.btn_trans_texture = QPushButton('Transfer Texture', self)
        self.btn_trans_texture.clicked[bool].connect(self.evt_trans_texture)
        self.grid2.addWidget(self.btn_trans_texture, 10, 1)
        self.btn_trans_texture.setEnabled(False)

        self.btn_trans_shape = QPushButton('Transfer Shape', self)
        self.btn_trans_shape.clicked[bool].connect(self.evt_trans_shape)
        self.grid2.addWidget(self.btn_trans_shape, 10, 2)
        self.btn_trans_shape.setEnabled(False)

        self.sld2val = {}
        self.val2sld = {}

        self.label_color = ['Color: Hue', 'Color: Saturation', 'Color: Brightness',
                            'Color: Variance']
        self.label_shape = ['Shape: Volume', 'Shape: Bangs', 'Shape: Length', 'Shape: Direction']
        self.label_curliness = ['Texture: Curliness']
        self.label_app = ['Texture: Smoothness', 'Texture: Thickness']
        self.label_total = self.label_color + self.label_shape + self.label_curliness + self.label_app

        col_num = 4
        row_num = 3
        for row in range(row_num):
            for col in range(col_num):
                if col == 3 and row == 2:
                    continue
                num = col_num * row + col
                sld = QSlider(Qt.Horizontal, self)
                sld.setMinimum(-self.maximum_value * 100)
                sld.setMaximum(self.maximum_value * 100)
                sld.sliderMoved[int].connect(self.evt_change_value)
                self.sld2val[sld] = num
                self.val2sld[num] = sld
                self.grid2.addWidget(QLabel(self.label_total[num]), row * 2 + 2, col)
                self.grid2.addWidget(sld, row * 2 + 2 + 1, col)
                sld.setEnabled(False)

        self.grid2.addWidget(QLabel(), 10, 3)

        whole_vbox = QVBoxLayout(self)
        whole_vbox.addLayout(self.grid1)
        whole_vbox.addLayout(self.grid2)

        self.setLayout(whole_vbox)
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('CtrlHair')
        self.show()

    def evt_open_target(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image file')
        if fname[0]:
            self.target_name = fname[0]
            self.load_target_image(fname[0])
            if self.input_name is not None:
                self.btn_trans_color.setEnabled(True)
                self.btn_trans_shape.setEnabled(True)
                self.btn_trans_texture.setEnabled(True)

    def evt_open_input(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image file')
        if fname[0]:
            input_name = fname[0]
            self.input_name = input_name
            self.load_input_image(input_name)
            self.btn_output.setEnabled(True)
            if self.target_name is not None:
                self.btn_trans_color.setEnabled(True)
                self.btn_trans_shape.setEnabled(True)
                self.btn_trans_texture.setEnabled(True)

            for kk in self.sld2val:
                kk.setEnabled(True)

    def evt_output(self):
        output_img = self.backend.output()
        img_path = os.path.join(self.temp_path, 'out_img.png')
        write_rgb(img_path, output_img)
        self.lbl_out_img.setPixmap((QPixmap(img_path)))

    def evt_trans_color(self):
        self.backend.transfer_latent_representation('color')
        self.refresh_slider()

    def evt_trans_texture(self):
        self.backend.transfer_latent_representation('texture')
        self.refresh_slider()

    def evt_trans_shape(self):
        self.backend.transfer_latent_representation('shape', refresh=True)
        self.refresh_slider()
        input_parsing_show = self.backend.get_cur_mask()
        input_parsing_path = os.path.join(self.temp_path, 'input_parsing.png')
        write_rgb(input_parsing_path, input_parsing_show)
        self.lbl_input_seg.setPixmap((QPixmap(input_parsing_path)))

    def load_input_image(self, img_path):
        img = read_rgb(img_path)
        if self.need_crop:
            img = self.backend.crop_face(img)
        input_img, input_parsing_show = self.backend.set_input_img(img_rgb=img)
        input_path = os.path.join(self.temp_path, 'input_img.png')
        write_rgb(input_path, input_img)
        self.lbl_input_img.setPixmap((QPixmap(input_path)))

        input_parsing_path = os.path.join(self.temp_path, 'input_parsing.png')
        write_rgb(input_parsing_path, input_parsing_show)
        self.lbl_input_seg.setPixmap((QPixmap(input_parsing_path)))
        self.refresh_slider()

        self.lbl_out_img.setPixmap(QPixmap(None))

    def load_target_image(self, img_path):
        img = read_rgb(img_path)
        if self.need_crop:
            img = self.backend.crop_face(img)
        input_img, input_parsing_show = self.backend.set_target_img(img_rgb=img)
        input_path = os.path.join(self.temp_path, 'target_img.png')
        write_rgb(input_path, input_img)
        self.lbl_target_img.setPixmap((QPixmap(input_path)))

    def refresh_slider(self):
        idx = 0
        # color
        color_val = self.backend.get_color_be2fe()
        for ii in range(4):
            self.val2sld[idx + ii].setValue(int(color_val[ii] * 100))

        # shape
        idx += len(self.label_color)
        shape_val = self.backend.get_shape_be2fe()
        for ii in range(4):
            self.val2sld[idx + ii].setValue(int(shape_val[ii] * 100))

        # curliness
        idx += len(self.label_shape)
        self.val2sld[idx].setValue(self.backend.get_curliness_be2fe() * 100)
        #  texture
        idx += len(self.label_curliness)
        app_val = self.backend.get_texture_be2fe()
        for ii in range(2):
            self.val2sld[idx + ii].setValue(int(app_val[ii] * 100))

    def evt_change_value(self, sld_v):
        """
        change all sliders value
        :param v: 0-100
        :return:
        """
        v = sld_v / 100.0
        sld_idx = self.sld2val[self.sender()]
        if sld_idx < len(self.label_color):
            self.backend.change_color(v, sld_idx)
            return
        sld_idx -= len(self.label_color)
        if sld_idx < len(self.label_shape):
            self.backend.change_shape(v, sld_idx)
            input_parsing_show = self.backend.get_cur_mask()
            input_parsing_path = os.path.join(self.temp_path, 'input_parsing.png')
            write_rgb(input_parsing_path, input_parsing_show)
            self.lbl_input_seg.setPixmap((QPixmap(input_parsing_path)))
            return
        sld_idx -= len(self.label_shape)
        if sld_idx < len(self.label_curliness):
            self.backend.change_curliness(v)
            return
        sld_idx -= len(self.label_curliness)
        if sld_idx < len(self.label_app):
            self.backend.change_texture(v, sld_idx)
            return


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
