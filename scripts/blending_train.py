import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Encoders import ClipBlendingModel as BlendingModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.train import toggle_grad, WandbLogger, image_grid, seed_everything, get_fid_calc


class Trainer:
    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 logger=None,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dilate_erosion = DilateErosion(device=self.device)

        if self.model is not None:
            self.fid_calc = get_fid_calc('input/fid.pkl', args.fid_dataset)

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt', channel_multiplier=2, latent=512,
                                 n_mlp=8, device=self.device))
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.device)
        self.seg.eval()

        self.seg.load_state_dict(torch.load('pretrained_models/BiSeNet/seg.pth'))
        toggle_grad(self.seg, False)
        toggle_grad(self.net.generator, False)

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_128 = BicubicDownSample(factor=8)

        self.best_loss = float('+inf')
        self.cur_iter = 0

    @torch.no_grad()
    def generate_mask(self, I):
        IM = (self.downsample_512((I + 1) / 2) - seg_mean) / seg_std
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

            torch.save({'model_state_dict': model_state_dict}, f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def calc_loss(self, I_gen, I_face, I_color, mask_face, mask_hair, gen_hair):
        gen_embed = self.model.get_image_embed(I_gen * mask_face)
        gt_embed = self.model.get_image_embed(I_face * mask_face)
        face_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        gen_embed = self.model.get_image_embed(I_gen * mask_hair)
        gt_embed = self.model.get_image_embed(I_color * mask_hair)
        hair_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        losses = {'face loss': face_loss, 'hair loss': hair_loss, 'loss': face_loss + hair_loss}
        return losses['loss'], losses

    def train_one_epoch(self):
        self.model.to(self.device).train()
        for batch in tqdm(self.train_dataloader):
            color_s, align_s, align_f, color_i, face_i, target_mask, HM_3E, HM_XE = map(lambda x: x.to(self.device),
                                                                                        batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), axis=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=align_f)

            loss, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_XE)

            self.optimizer.zero_grad()
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.logger.next_step()
            for key, val in info.items():
                self.logger.log(key, val.item())
            self.logger.log('grad', total_norm.item())
            self.cur_iter += 1

    @torch.no_grad()
    def validate(self):
        self.model.to(self.device).eval()

        sum_losses = lambda x, y: {key: val + x.get(key, 0) for key, val in y.items()}
        files = []
        losses = {}
        to_299 = T.Resize((299, 299))
        images_to_fid = []

        for batch in tqdm(self.test_dataloader):
            color_s, align_s, align_f, color_i, face_i, target_mask, HM_3E, HM_XE = map(lambda x: x.to(self.device),
                                                                                        batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), axis=1)
            I_G, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=align_f)

            _, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_XE)
            losses = sum_losses(losses, info)
            for k in range(bsz):
                files.append([color_i[k].cpu(), face_i[k].cpu(), self.downsample_256(I_G)[k].cpu()])

            images_to_fid.append(to_299((I_G + 1) / 2).clip(0, 1))

        losses['FID CLIP'] = self.fid_calc(torch.cat(images_to_fid))
        for key, val in losses.items():
            if key != 'FID CLIP':
                val = val.item() / len(self.test_dataloader)
            self.logger.log(f'val {key}', val)

        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=100, replace=False)
        images_to_log = [
            image_grid([T.functional.to_pil_image(((img + 1) / 2).clamp(0, 1)) for img in files[idx]], 1, 3) for idx in
            idxs]
        self.logger.log('val images', [wandb.Image(image) for image in images_to_log])

        return losses['loss']

    def train_loop(self, epochs):
        self.validate()
        for epoch in range(epochs):
            self.train_one_epoch()
            loss = self.validate()

            self.save_model('last', save_online=False)
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f'best', save_online=False)


def prepare_item(exp, path):
    im1, im2, im3 = exp

    try:
        color_path = os.path.join(path, 'FS', f'{im3}.npz')
        Color_S = torch.from_numpy(np.load(color_path)['latent_in']).squeeze(0)

        face_path = os.path.join(path, 'FS', f'{im1}.npz')
        Align_S = torch.from_numpy(np.load(face_path)['latent_in']).squeeze(0)

        Color_I = T.functional.normalize(T.functional.to_tensor(
            Image.open(os.path.join(args.FFHQ, f'{im3}.png'))
        ), [0.5], [0.5])
        Face_I = T.functional.normalize(T.functional.to_tensor(
            Image.open(os.path.join(args.FFHQ, f'{im1}.png'))
        ), [0.5], [0.5])

        align_path = os.path.join(path, 'Align')
        data = np.load(
            os.path.join(align_path, f'{im1}_{im3}.npz')
        )
        Align_F = torch.from_numpy(data['latent_F']).squeeze(0)

        return (Color_S, Align_S, Align_F, Color_I, Face_I)
    except Exception as e:
        print(e, file=sys.stderr)
        return None


class Blending_dataset(Dataset):
    def __init__(self, exps, path, net_trainer):
        super().__init__()
        downsample_256 = BicubicDownSample(factor=4)
        data = Parallel(n_jobs=-1)(
            delayed(prepare_item)(exp, path) for (p1, p2, p3) in tqdm(exps) for exp in [(p1, p2, p3), (p1, p3, p2)])
        data = [elem for elem in data if elem is not None]
        print(f'Load: {len(data)}/{2 * len(exps)}', file=sys.stderr)

        tmp_dataloader = DataLoader(data, batch_size=24, pin_memory=False, shuffle=False)

        self.items = []
        with torch.no_grad():
            for (Color_S, Align_S, Align_F, Color_I, Face_I) in tqdm(tmp_dataloader):
                HM_3D, HM_3E = net_trainer.generate_mask(Color_I.to('cuda'))

                HM_1D, _ = net_trainer.generate_mask(Face_I.to('cuda'))
                I_X, _ = net_trainer.net.generator([Align_S.to('cuda')], input_is_latent=True, return_latents=False,
                                                   start_layer=4,
                                                   end_layer=8, layer_in=Align_F.to('cuda'))
                HM_XD, HM_XE = net_trainer.generate_mask(I_X)

                target_mask = ((1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)).cpu()
                HM_3E = HM_3E.cpu()
                HM_XE = HM_XE
                self.items.extend(
                    [item for item in zip(*list(map(lambda x: [item.squeeze(0) for item in torch.split(x, 1)],
                                                    (Color_S,
                                                     Align_S,
                                                     Align_F,
                                                     downsample_256(Color_I.to('cuda')).cpu(),
                                                     downsample_256(Face_I.to('cuda')).cpu(),
                                                     target_mask, HM_3E, HM_XE)))
                                          ) if item[-2].any() and item[-1].any()]
                )

        print(f'dataset: {len(self.items)}/{len(data)}', file=sys.stderr)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main(args):
    seed_everything()

    exps = []
    with open(os.path.join(args.dataset, 'dataset.exps'), 'r') as file:
        for exp in file.readlines():
            exps.append(list(map(lambda x: x.replace('.png', ''), exp.split())))

    X_train, X_test = train_test_split(exps, test_size=512, random_state=42)

    net_trainer = Trainer()
    train_dataset = Blending_dataset(X_train, args.dataset, net_trainer)
    test_dataset = Blending_dataset(X_test, args.dataset, net_trainer)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger = WandbLogger(name=args.name_run, project='Barbershop-Blending')
    logger.start_logging()
    logger.save(__file__)

    model = BlendingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.000001)

    trainer = Trainer(model, optimizer, None, train_dataloader, test_dataloader, logger)

    trainer.train_loop(1000)

    logger.wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blending trainer')
    parser.add_argument('--name_run', type=str, default='test')
    parser.add_argument('--dataset', type=Path, default='input/blending_dataset')
    parser.add_argument('--FFHQ', type=Path)
    parser.add_argument('--fid_dataset', type=str, default='input')
    args = parser.parse_args()

    main(args)
