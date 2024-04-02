import argparse
import os
import sys
from argparse import Namespace
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.STAR.lib import utility
from models.Encoders import RotateModel
from models.Net import Net
from models.Net import iresnet100
from models.encoder4editing.utils.model_utils import setup_model, get_latents
from utils.bicubic import BicubicDownSample
from utils.train import image_grid, WandbLogger, seed_everything, toggle_grad


class MovingAverageLoss:
    def __init__(self, weights: dict, alpha=0.02):
        self.alpha = alpha
        self.weights = weights
        self.vals = {}

    def reset(self):
        self.vals = {}

    def update(self, cur_vals):
        for key, val in cur_vals.items():
            self.vals[key] = self.alpha * val + (1 - self.alpha) * self.vals.get(key, val)

    def calc_loss(self, losses):
        loss = 0.
        for key, val in losses.items():
            loss += self.weights.get(key, 1) * val / self.vals.get(key, 1)
        return loss


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

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/StyleGAN/ffhq.pt', channel_multiplier=2, latent=512,
                                 n_mlp=8, device=self.device))
        self.e4e = setup_model('pretrained_models/encoder4editing/e4e_ffhq_encode.pt', 'cuda')[0]
        self.arc_face = iresnet100()
        self.arc_face.load_state_dict(torch.load("pretrained_models/ArcFace/backbone_r100.pth"))
        self.arc_face.eval().cuda()
        self.toArcface = T.Compose([
            T.Resize((112, 112)),
            T.Normalize(0.5, 0.5)
        ])

        # init landmarks
        config = utility.get_config(utility.landmarks_arg)
        self.kp_extractor = utility.get_net(config)
        model_path = utility.landmarks_arg.pretrained_weight
        checkpoint = torch.load(model_path)
        self.kp_extractor.load_state_dict(checkpoint["net"])
        self.kp_extractor = self.kp_extractor.float().to('cuda')
        self.kp_extractor.eval()
        self.toLandmarks = T.Compose([
            T.Resize((256, 256)),
            T.Normalize(0.5, 0.5)
        ])

        toggle_grad(self.arc_face, False)
        toggle_grad(self.kp_extractor, False)
        toggle_grad(self.net.generator, False)
        toggle_grad(self.e4e.encoder, False)

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_128 = BicubicDownSample(factor=8)

        self.MAL = MovingAverageLoss({'mse points to': 6, 'mse latents': 2})
        self.best_loss = float('+inf')

    def generate_key_points(self, batch):
        _, _, landmarks = self.kp_extractor(self.toLandmarks(batch))
        final_marks_2D = (landmarks[:, :76] + 1) / 2 * torch.tensor([256 - 1, 256 - 1]).to('cuda').view(1, 1, 2)
        return final_marks_2D

    @torch.no_grad()
    def generate_latents(self, batch):
        return get_latents(self.e4e, batch)

    def save_model(self, name, save_online=True):
        with TemporaryDirectory() as tmp_dir:
            model_state_dict = self.model.state_dict()

            # delete pretrained clip
            for key in list(model_state_dict.keys()):
                if key.startswith("clip_model."):
                    del model_state_dict[key]

            torch.save({'model_state_dict': model_state_dict}, f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def load_model(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)

    def calc_loss(self,
                  I_to,
                  I_from,
                  key_points_to,
                  latents_from,
                  latents_to,
                  ret_images=False,
                  normalize=True
                  ):
        # rotate
        rotate_to = self.model(latents_from[:, :6], latents_to[:, :6])
        latent_in = torch.cat((rotate_to, latents_from[:, 6:]), axis=1)
        I_G_to, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
        I_G_to_0_1 = ((I_G_to + 1) / 2)
        I_gen_to = self.downsample_256(I_G_to_0_1).clip(0, 1)

        # key_point_loss
        key_points_gen_to = self.generate_key_points(I_gen_to)
        key_point_loss_to = F.mse_loss(key_points_gen_to, key_points_to)

        # arcface loss
        gen_embed = self.arc_face(self.toArcface(I_gen_to))
        gt_embed = self.arc_face(self.toArcface(I_from))
        arc_face_loss = 20 * (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        losses = {
            'mse points to': key_point_loss_to,
            'arc face': arc_face_loss
        }

        if normalize:
            losses['loss'] = self.MAL.calc_loss(losses)
        else:
            losses['loss'] = sum(losses.values())

        if ret_images:
            return losses['loss'], {key: val.item() for key, val in losses.items()}, I_gen_to, latent_in
        else:
            return losses['loss'], {key: val.item() for key, val in losses.items()}

    def calc_hair_loss(self,
                       latents_from,
                       latents_to,
                       ret_images=False,
                       normalize=True
                       ):
        # rotate
        rotate_to = self.model(latents_from[:, :6], latents_to[:, :6])
        mse_latents = 300 * F.mse_loss(rotate_to, latents_to[:, :6])

        losses = {
            'mse latents': mse_latents
        }

        if normalize:
            losses['loss'] = self.MAL.calc_loss(losses)
        else:
            losses['loss'] = sum(losses.values())

        if ret_images:
            latent_in = torch.cat((rotate_to, latents_from[:, 6:]), axis=1)
            I_G_to, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
            I_G_to_0_1 = ((I_G_to + 1) / 2)
            I_gen_to = self.downsample_256(I_G_to_0_1).clip(0, 1)

            return losses['loss'], {key: val.item() for key, val in losses.items()}, I_gen_to
        else:
            return losses['loss'], {key: val.item() for key, val in losses.items()}

    def train_one_epoch(self):
        self.model.to(self.device).train()
        sum_losses = lambda x, y: {key: y.get(key, 0) + x.get(key, 0) for key in set(x.keys()) | set(y.keys())}

        dataloader_to = iter(self.train_dataloader)
        for batch in tqdm(self.train_dataloader):
            I_from, key_points_from, latents_from = map(lambda x: x.to(self.device), batch)
            I_to, key_points_to, latents_to = map(lambda x: x.to(self.device), next(dataloader_to))

            self.optimizer.zero_grad()

            loss, info, _, gen_latent = self.calc_loss(
                I_to,
                I_from,
                key_points_to,
                latents_from,
                latents_to,
                ret_images=True
            )

            if self.args.use_hair_loss:
                hair_loss, info2 = self.calc_hair_loss(
                    gen_latent,
                    latents_from
                )
                loss += hair_loss
                info = sum_losses(info, info2)
            loss.backward()

            self.MAL.update(info)

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.logger.next_step()
            for key, val in info.items():
                self.logger.log(key, val)
            self.logger.log('grad', total_norm.item())

    @torch.no_grad()
    def validate(self):
        self.model.to(self.device).eval()

        sum_losses = lambda x, y: {key: y.get(key, 0) + x.get(key, 0) for key in set(x.keys()) | set(y.keys())}
        files = []
        losses = {}

        for batch in tqdm(self.test_dataloader):
            I_from, key_points_from, latents_from, \
                I_to, key_points_to, latents_to, = map(lambda x: x.to(self.device), batch)
            bsz = I_from.size(0)

            loss, info, I_gen_to, gen_latent = self.calc_loss(
                I_to,
                I_from,
                key_points_to,
                latents_from,
                latents_to,
                ret_images=True,
                normalize=False
            )

            if args.use_hair_loss:
                loss, info2, I_gen_to_rec = self.calc_hair_loss(
                    gen_latent,
                    latents_from,
                    ret_images=True,
                    normalize=False
                )
                losses = sum_losses(losses, info2)
            else:
                I_G_from, _ = self.net.generator([latents_from], input_is_latent=True, return_latents=False)
                I_G_from_0_1 = ((I_G_from + 1) / 2)
                I_gen_to_rec = self.downsample_256(I_G_from_0_1).clip(0, 1)

            losses = sum_losses(losses, info)
            for k in range(bsz):
                files.append([I_from[k].cpu(), I_gen_to_rec[k].cpu(), I_gen_to[k].cpu(), I_to[k].cpu()])

        for key, val in losses.items():
            val /= len(self.test_dataloader)
            self.logger.log(f'val {key}', val)

        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=min(len(files), 100), replace=False)
        images_to_log = [image_grid(list(map(T.functional.to_pil_image, files[idx])), 1, 4) for idx in idxs]
        self.logger.log('val images', [wandb.Image(image) for image in images_to_log])

        return losses['loss'] / len(self.test_dataloader)

    def train_loop(self, epochs):
        # self.validate()
        for epoch in range(epochs):
            self.train_one_epoch()
            loss = self.validate()

            self.save_model(f'rotate_{epoch}', save_online=False)
            self.save_model('last')
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f'best', save_online=False)


class Rotate_dataset(Dataset):
    def __init__(self, tensors_images, key_points, latents, is_test=False):
        super().__init__()
        self.tensors_images = tensors_images
        self.key_points = key_points
        self.latents = latents
        self.is_test = is_test

    def __len__(self):
        return len(self.tensors_images)

    def __get_elem__(self, idx):
        return self.tensors_images[idx], self.key_points[idx], self.latents[idx]

    def __getitem__(self, idx):
        if self.is_test:
            return *self.__get_elem__(idx), *self.__get_elem__(-idx)
        else:
            return self.__get_elem__(idx)


def main(args):
    seed_everything()
    data = list(torch.load(args.dataset).values())

    X_train, X_test = train_test_split(list(zip(data[0], data[1], data[2])), test_size=512, random_state=42)

    train_dataset = Rotate_dataset(*list(zip(*X_train)))
    test_dataset = Rotate_dataset(*list(zip(*X_test)), is_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                                  drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False,
                                 num_workers=4)

    logger = WandbLogger(name=args.name_run, project='HairFast-Rotate')
    logger.start_logging()
    logger.save(__file__)

    model = RotateModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.000001)

    trainer = Trainer(model, args, optimizer, None, train_dataloader, test_dataloader, logger)
    trainer.train_loop(1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotate trainer')
    parser.add_argument('--name_run', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='input/rotate_dataset.pkl')
    parser.add_argument('--use_hair_loss', action='store_false')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    main(args)
