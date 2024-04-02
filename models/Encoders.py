import argparse

import clip
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential
from torchvision import transforms as T

from models.Net import FeatureEncoderMult, IBasicBlock, conv1x1
from models.stylegan2.model import PixelNorm


class ModulationModule(nn.Module):
    def __init__(self, layernum, last=False, inp=512, middle=512):
        super().__init__()
        self.layernum = layernum
        self.last = last
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.beta_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        gamma = self.gamma_function(embedding)
        beta = self.beta_function(embedding)
        out = x * (1 + gamma) + beta
        if not self.last:
            out = self.leakyrelu(out)
        return out


class FeatureiResnet(nn.Module):
    def __init__(self, blocks, inplanes=1024):
        super().__init__()

        self.res_blocks = {}

        for n, block in enumerate(blocks, start=1):
            planes, num_blocks = block

            for k in range(1, num_blocks + 1):
                downsample = None
                if inplanes != planes:
                    downsample = nn.Sequential(conv1x1(inplanes, planes, 1), nn.BatchNorm2d(planes, eps=1e-05, ), )

                self.res_blocks[f'res_block_{n}_{k}'] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
                inplanes = planes

        self.res_blocks = nn.ModuleDict(self.res_blocks)

    def forward(self, x):
        for module in self.res_blocks.values():
            x = module(x)
        return x


class RotateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(6, i == 4) for i in range(5)])

    def forward(self, latent_from, latent_to):
        dt_latent = self.pixelnorm(latent_from)
        for modulation_module in self.modulation_module_list:
            dt_latent = modulation_module(dt_latent, latent_to)
        output = latent_from + 0.1 * dt_latent
        return output


class ClipBlendingModel(nn.Module):
    def __init__(self, clip_model="ViT-B/32"):
        super().__init__()
        self.pixelnorm = PixelNorm()
        self.clip_model, _ = clip.load(clip_model, device="cuda")
        self.transform = T.Compose(
            [T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.modulation_module_list = nn.ModuleList(
            [ModulationModule(12, i == 4, inp=512 * 3, middle=1024) for i in range(5)]
        )

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def get_image_embed(self, image_tensor):
        resized_tensor = self.face_pool(image_tensor)
        renormed_tensor = self.transform(resized_tensor * 0.5 + 0.5)
        return self.clip_model.encode_image(renormed_tensor)

    def forward(self, latent_face, latent_color, target_face, hair_color):
        embed_face = self.get_image_embed(target_face).unsqueeze(1).expand(-1, 12, -1)
        embed_color = self.get_image_embed(hair_color).unsqueeze(1).expand(-1, 12, -1)
        latent_in = torch.cat((latent_color, embed_face, embed_color), dim=-1)

        dt_latent = self.pixelnorm(latent_face)
        for modulation_module in self.modulation_module_list:
            dt_latent = modulation_module(dt_latent, latent_in)
        output = latent_face + 0.1 * dt_latent
        return output


class PostProcessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': "pretrained_models/ArcFace/backbone_ir50.pth"}))

        self.latent_avg = torch.load('pretrained_models/PostProcess/latent_avg.pt', map_location=torch.device('cuda'))
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.pixelnorm = PixelNorm()

    def forward(self, source, target):
        s_face, [f_face] = self.encoder_face(source)
        s_hair, [f_hair] = self.encoder_face(target)

        dt_latent_face = self.pixelnorm(s_face)
        dt_latent_hair = self.pixelnorm(s_hair)

        for mod_module in self.to_latent_1:
            dt_latent_face = mod_module(dt_latent_face, s_hair)

        for mod_module in self.to_latent_2:
            dt_latent_hair = mod_module(dt_latent_hair, s_face)

        finall_s = self.latent_avg + 0.1 * (dt_latent_face + dt_latent_hair)

        cat_f = torch.cat((f_face, f_hair), dim=1)
        finall_f = self.to_feature(cat_f)

        return finall_s, finall_f


class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.transform = T.Compose(
            [T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image_tensor):
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.to("cuda")
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor / 255

        resized_tensor = self.face_pool(image_tensor)
        renormed_tensor = self.transform(resized_tensor)
        return self.clip_model.encode_image(renormed_tensor)
