import argparse
import os
import random
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from losses.pp_losses import LossBuilder, LossBuilderMulti
from models.Encoders import ModulationModule, FeatureiResnet, FeatureEncoderMult
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.stylegan2 import dnnlib
from models.stylegan2.model import PixelNorm
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import image_grid, WandbLogger, toggle_grad, _LegacyUnpickler, seed_everything, get_fid_calc


class Trainer:
    def __init__(self,
                 model=None,
                 args=None,
                 optimizer=None,
                 scheduler=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 logger=None
                 ):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dilate_erosion = DilateErosion(device=self.device)
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.model is not None:
            self.fid_calc = get_fid_calc('input/fid.pkl', args.fid_dataset)

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt', channel_multiplier=2, latent=512,
                                 n_mlp=8, device=self.device))

        with dnnlib.util.open_url("pretrained_models/StyleGAN/ffhq.pkl") as f:
            data = _LegacyUnpickler(f).load()
        self.discriminator = data['D'].cuda().eval()
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4, betas=(0.9, 0.999), amsgrad=False,
                                           weight_decay=0)

        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.device)
        self.seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))
        self.seg.eval()

        toggle_grad(self.discriminator, False)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.seg, False)

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_128 = BicubicDownSample(factor=8)

        self.best_loss = float('+inf')
        if self.args is not None:
            if self.args.pretrain:
                self.LossBuilder = LossBuilder(
                    {'lpips_scale': 0.8, 'id': 0.1, 'landmark': 0, 'feat_rec': 0.01, 'adv': self.args.adv_coef})
            else:
                self.LossBuilder = LossBuilderMulti(
                    {'lpips_scale': 0.8, 'id': 0.1, 'landmark': 0.1, 'feat_rec': 0.01, 'adv': self.args.adv_coef,
                     'inpaint': self.args.inpaint})
        self.cur_iter = 1

    @torch.no_grad()
    def generate_mask(self, I):
        IM = (self.downsample_512(I) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        HM_X = F.interpolate(HM_X.unsqueeze(1), size=(256, 256), mode='nearest')

        HM_XD, HM_XE = self.dilate_erosion.mask(HM_X)
        return HM_XD, HM_XE

    def save_model(self, name, save_online=True):
        with TemporaryDirectory() as tmp_dir:
            model_state_dict = self.model.state_dict()

            # delete pretrained clip
            for key in list(model_state_dict.keys()):
                if key.startswith("clip_model."):
                    del model_state_dict[key]

            torch.save(
                {'model_state_dict': model_state_dict, 'D': self.discriminator.state_dict(), 'cur_iter': self.cur_iter},
                f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'D' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['D'], strict=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def train_one_epoch(self):
        self.model.to(self.device).train()
        for batch in tqdm(self.train_dataloader):
            source, target, target_mask, HT_E = map(lambda x: x.to(self.device), batch)
            source, source_1024 = self.downsample_256(source).clip(0, 1), self.normalize(source)

            latent_s, latent_f = self.model(self.normalize(source), self.normalize(target), target_mask, HT_E)

            gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
            F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=0, end_layer=4)

            if self.args.pretrain:
                alpha = min(1, self.cur_iter / self.args.iter_before)
                latent_f_gen = alpha * latent_f + (1 - alpha) * F_w
            else:
                latent_f_gen = latent_f

            gen_im_F, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                             start_layer=5, end_layer=8, layer_in=latent_f_gen)

            losses = self.LossBuilder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f)

            if self.args.use_adv and self.cur_iter >= self.args.iter_before:
                losses.update(self.LossBuilder.CalcAdvLoss(self.discriminator, gen_im_F))

            losses['loss'] = sum(losses.values())

            self.optimizer.zero_grad()
            losses['loss'].backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if self.args.use_adv and self.cur_iter >= self.args.iter_before:
                if self.cur_iter == self.args.iter_before:
                    print('Start scripts discr')

                toggle_grad(self.discriminator, True)
                self.discriminator.train()

                disc_loss = self.LossBuilder.CalcDisLoss(self.discriminator, source_1024, gen_im_F.detach())
                if self.cur_iter % self.args.d_reg_every:
                    disc_loss.update(self.LossBuilder.CalcR1Loss(self.discriminator, source_1024))

                total_loss = sum(disc_loss.values())

                self.disc_optim.zero_grad()

                total_loss.backward()
                total_norm_d = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                disc_loss['grad disc'] = total_norm_d

                self.disc_optim.step()

                toggle_grad(self.discriminator, False)
                self.discriminator.eval()
                losses.update(disc_loss)

            losses['scripts grad'] = total_norm
            self.logger.next_step()
            self.logger.log_scalars({f'scripts {key}': val for key, val in losses.items()})
            self.cur_iter += 1

    @torch.no_grad()
    def validate(self):
        self.model.to(self.device).eval()

        sum_losses = lambda x, y: {key: y.get(key, 0) + x.get(key, 0) for key in set(x.keys()) | set(y.keys())}
        files = []
        val_losses = {}
        to_299 = T.Resize((299, 299))
        images_to_fid = []

        for batch in tqdm(self.test_dataloader):
            source, target, target_mask, HT_E = map(lambda x: x.to(self.device), batch)
            source = self.downsample_256(source).clip(0, 1)
            bsz = source.size(0)

            latent_s, latent_f = self.model(self.normalize(source), self.normalize(target), target_mask, HT_E)

            gen_im_W, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False)
            F_w, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                        start_layer=0, end_layer=4)
            gen_im_F, _ = self.net.generator([latent_s], input_is_latent=True, return_latents=False,
                                             start_layer=5, end_layer=8, layer_in=latent_f)

            losses = self.LossBuilder(source, target, target_mask, HT_E, gen_im_W, F_w, gen_im_F, latent_f)
            losses['loss'] = sum(losses.values())

            gen_w_256 = self.downsample_256((gen_im_W + 1) / 2).clip(0, 1)
            gen_f_256 = self.downsample_256((gen_im_F + 1) / 2).clip(0, 1)

            images_to_fid.append(to_299((gen_im_F + 1) / 2).clip(0, 1))

            val_losses = sum_losses(val_losses, losses)
            for k in range(bsz):
                files.append([source[k].cpu(), target[k].cpu(), gen_w_256[k].cpu(), gen_f_256[k].cpu()])

        val_losses['FID CLIP'] = self.fid_calc(torch.cat(images_to_fid))
        for key, val in val_losses.items():
            if key != 'FID CLIP':
                val = val.item() / len(self.test_dataloader)
            self.logger.log_scalars({f'val {key}': val})

        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=min(len(files), 100), replace=False)
        images_to_log = [image_grid(list(map(T.functional.to_pil_image, files[idx])), 1, len(files[idx])) for idx in
                         idxs]
        self.logger.log_scalars({'val images': [wandb.Image(image) for image in images_to_log]})

        return val_losses['loss']

    def train_loop(self, epochs):
        self.validate()
        for epoch in range(epochs):
            self.train_one_epoch()
            loss = self.validate()

            self.save_model('last', save_online=False)
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f'best_{epoch}', save_online=False)


class PP_dataset(Dataset):
    def __init__(self, source, target, target_mask, HT_E, is_test=False):
        super().__init__()
        self.source = source
        self.target = target
        self.target_mask = target_mask
        self.HT_E = HT_E
        self.is_test = is_test

    def __len__(self):
        return len(self.source)

    def load_image(self, path):
        return T.functional.to_tensor(Image.open(path))

    def __transform__(self, img1, img2, mask1, mask2):
        if self.is_test:
            return img1, img2, mask1, mask2

        if random.random() > 0.5:
            img1 = T.functional.hflip(img1)
            img2 = T.functional.hflip(img2)
            mask1 = T.functional.hflip(mask1)
            mask2 = T.functional.hflip(mask2)

        return img1, img2, mask1, mask2

    def __getitem__(self, idx):
        return self.__transform__(self.load_image(self.source[idx]), self.target[idx], self.target_mask[idx],
                                  self.HT_E[idx])


class PostProcessModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': "pretrained_models/ArcFace/backbone_ir50.pth"}))
        if not self.args.finetune:
            toggle_grad(self.encoder_face, False)

        self.latent_avg = torch.load('pretrained_models/PostProcess/latent_avg.pt', map_location=torch.device('cuda'))
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        if self.args.use_mod:
            self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.pixelnorm = PixelNorm()
        else:
            self.to_latent = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm([1024]), nn.LeakyReLU(),
                                           nn.Linear(1024, 512))

    def forward(self, source, target, target_mask=None, *args, **kwargs):
        s_face, [f_face] = self.encoder_face(source)
        if self.args.pretrain:
            return self.latent_avg + s_face, f_face

        s_hair, [f_hair] = self.encoder_face(target)

        if self.args.use_mod:
            dt_latent_face = self.pixelnorm(s_face)
            dt_latent_hair = self.pixelnorm(s_hair)

            for mod_module in self.to_latent_1:
                dt_latent_face = mod_module(dt_latent_face, s_hair)

            for mod_module in self.to_latent_2:
                dt_latent_hair = mod_module(dt_latent_hair, s_face)
            finall_s = self.latent_avg + 0.1 * (dt_latent_face + dt_latent_hair)
        else:
            cat_s = torch.cat((s_face, s_hair), dim=-1)
            finall_s = self.latent_avg + self.to_latent(cat_s)

        if self.args.use_full:
            cat_f = torch.cat((f_face, f_hair), dim=1)
        else:
            t_mask = F.interpolate(target_mask, size=(64, 64), mode='nearest')
            cat_f = torch.cat((f_face * t_mask, f_hair * (1 - t_mask)), dim=1)

        finall_f = self.to_feature(cat_f)
        return finall_s, finall_f


def main(args):
    seed_everything()
    dataset = []

    idx = 1
    while os.path.isfile(args.dataset / f'pp_part_{idx}.dataset'):
        batch_data = torch.load(args.dataset / f'pp_part_{idx}.dataset')
        dataset.extend(batch_data)
        idx += 1

    X_train, X_test = train_test_split(dataset, test_size=1024, random_state=42)

    train_dataset = PP_dataset(*list(zip(*X_train)))
    test_dataset = PP_dataset(*list(zip(*X_test)), is_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=10, pin_memory=True,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=10, pin_memory=True,
                                 shuffle=False)

    logger = WandbLogger(name=args.name_run, project='HairFast-PostProcess')
    logger.start_logging()
    logger.save(__file__)

    model = PostProcessModel(args)
    if args.pretrain:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

    trainer = Trainer(model, args, optimizer, None, train_dataloader, test_dataloader, logger)
    if not args.pretrain:
        trainer.load_model(args.checkpoint)
    trainer.train_loop(1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post Process trainer')
    parser.add_argument('--name_run', type=str, default='test')
    parser.add_argument('--FFHQ', type=Path)
    parser.add_argument('--dataset', type=Path, default='input/pp_dataset')
    parser.add_argument('--fid_dataset', type=str, default='input')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iter_before', type=int, default=10_000)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--inpaint', type=float, default=0.)
    parser.add_argument('--use_adv', action='store_true')
    parser.add_argument('--adv_coef', type=float, default=0.05)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--use_mod', action='store_true')
    parser.add_argument('--use_full', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    main(args)
