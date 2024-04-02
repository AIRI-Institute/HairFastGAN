import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor
from models.stylegan2.model import Generator
from utils.drive import download_weight

transform_to_256 = transforms.Compose([
    transforms.Resize((256, 256)),
])

__all__ = ['Net', 'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200', 'FeatureEncoderMult',
           'IBasicBlock', 'conv1x1', 'get_segmentation']


class Net(nn.Module):

    def __init__(self, opts):
        super(Net, self).__init__()
        self.opts = opts
        self.generator = Generator(opts.size, opts.latent, opts.n_mlp, channel_multiplier=opts.channel_multiplier)
        self.cal_layer_num()
        self.load_weights()
        self.load_PCA_model()
        FaceParsing_tensor.parsing_img()

    def load_weights(self):
        if not os.path.exists(self.opts.ckpt):
            print('Downloading StyleGAN2 checkpoint: {}'.format(self.opts.ckpt))
            download_weight(self.opts.ckpt)

        print('Loading StyleGAN2 from checkpoint: {}'.format(self.opts.ckpt))
        checkpoint = torch.load(self.opts.ckpt)
        device = self.opts.device
        self.generator.load_state_dict(checkpoint['g_ema'])
        self.latent_avg = checkpoint['latent_avg']
        self.generator.to(device)
        self.latent_avg = self.latent_avg.to(device)

        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()

    def build_PCA_model(self, PCA_path):

        with torch.no_grad():
            latent = torch.randn((1000000, 512), dtype=torch.float32)
            # latent = torch.randn((10000, 512), dtype=torch.float32)
            self.generator.style.cpu()
            pulse_space = torch.nn.LeakyReLU(5)(self.generator.style(latent)).numpy()
            self.generator.style.to(self.opts.device)

        from utils.PCA_utils import IPCAEstimator

        transformer = IPCAEstimator(512)
        X_mean = pulse_space.mean(0)
        transformer.fit(pulse_space - X_mean)
        X_comp, X_stdev, X_var_ratio = transformer.get_components()
        np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)

    def load_PCA_model(self):
        device = self.opts.device

        PCA_path = self.opts.ckpt[:-3] + '_PCA.npz'

        if not os.path.isfile(PCA_path):
            self.build_PCA_model(PCA_path)

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)

    # def make_noise(self):
    #     noises_single = self.generator.make_noise()
    #     noises = []
    #     for noise in noises_single:
    #         noises.append(noise.repeat(1, 1, 1, 1).normal_())
    #
    #     return noises

    def cal_layer_num(self):
        if self.opts.size == 1024:
            self.layer_num = 18
        elif self.opts.size == 512:
            self.layer_num = 16
        elif self.opts.size == 256:
            self.layer_num = 14

        self.S_index = self.layer_num - 11

        return

    def cal_p_norm_loss(self, latent_in):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = self.opts.p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss

    def cal_l_F(self, latent_F, F_init):
        return self.opts.l_F_lambda * (latent_F - F_init).pow(2).mean()


def get_segmentation(img_rgb, resize=True):
    parsing, _ = FaceParsing_tensor.parsing_img(img_rgb)
    parsing = FaceParsing_tensor.swap_parsing_label_to_celeba_mask(parsing)
    mask_img = parsing.long()[None, None, ...]
    if resize:
        mask_img = transforms.functional.resize(mask_img, (256, 256),
                                                interpolation=transforms.InterpolationMode.NEAREST)
    return mask_img


fs_kernals = {
    0: (12, 12),
    1: (12, 12),
    2: (6, 6),
    3: (6, 6),
    4: (3, 3),
    5: (3, 3),
    6: (3, 3),
    7: (3, 3),
}

fs_strides = {
    0: (7, 7),
    1: (7, 7),
    2: (4, 4),
    3: (4, 4),
    4: (2, 2),
    5: (2, 2),
    6: (1, 1),
    7: (1, 1),
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05, )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = []
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            out.append(x)
            x = self.layer2(x)
            out.append(x)
            x = self.layer3(x)
            out.append(x)
            x = self.layer4(x)
            out.append(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)

        if return_features:
            out.append(x)
            return out
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)


class FeatureEncoder(nn.Module):
    def __init__(self, n_styles=18, opts=None, residual=False,
                 use_coeff=False, resnet_layer=None,
                 video_input=False, f_maps=512, stride=(1, 1)):
        super(FeatureEncoder, self).__init__()

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        # input conv layer
        if video_input:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                *list(resnet50.children())[1:3]
            )
        else:
            self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # define layers
        self.block_1 = list(resnet50.children())[3]  # 15-18
        self.block_2 = list(resnet50.children())[4]  # 10-14
        self.block_3 = list(resnet50.children())[5]  # 5-9
        self.block_4 = list(resnet50.children())[6]  # 1-4
        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

    def apply_head(self, x):
        latents = []
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)
        return out

    def forward(self, x):
        latents = []
        features = []
        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        content = self.content_layer(x)
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        return self.apply_head(x), content


class FeatureEncoderMult(FeatureEncoder):
    def __init__(self, fs_layers=(5,), ranks=None, **kwargs):
        super().__init__(**kwargs)

        self.fs_layers = fs_layers
        self.content_layer = nn.ModuleList()
        self.ranks = ranks
        shift = 0 if max(fs_layers) <= 7 else 2
        scale = 1 if max(fs_layers) <= 7 else 2
        for i in range(len(fs_layers)):
            if ranks is not None:
                stride, kern = ranks_data[ranks[i] - shift]
                layer1 = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(fs_kernals[fs_layers[i] - shift][0], kern),
                              stride=(fs_strides[fs_layers[i] - shift][0], stride),
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                layer2 = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                              bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(kern, fs_kernals[fs_layers[i] - shift][1]),
                              stride=(stride, fs_strides[fs_layers[i] - shift][1]),
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True)
                )
                layer = nn.ModuleList([layer1, layer2])
            else:
                layer = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=fs_kernals[fs_layers[i] - shift],
                              stride=fs_strides[fs_layers[i] - shift],
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
            self.content_layer.append(layer)

    def forward(self, x):
        x = transform_to_256(x)
        features = []
        content = []
        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        if max(self.fs_layers) > 7:
            for layer in self.content_layer:
                if self.ranks is not None:
                    mat1 = layer[0](x)
                    mat2 = layer[1](x)
                    content.append(torch.matmul(mat1, mat2))
                else:
                    content.append(layer(x))
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        if len(content) == 0:
            for layer in self.content_layer:
                if self.ranks is not None:
                    mat1 = layer[0](x)
                    mat2 = layer[1](x)
                    content.append(torch.matmul(mat1, mat2))
                else:
                    content.append(layer(x))
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        return self.apply_head(x), content


def get_keys(d, name, key="state_dict"):
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[: len(name) + 1] == name + '.'}
    return d_filt
