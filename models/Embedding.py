from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader

from datasets.image_dataset import ImagesDataset, image_collate
from models.FeatureStyleEncoder import FSencoder
from models.Net import Net, get_segmentation
from models.encoder4editing.utils.model_utils import setup_model, get_latents
from utils.bicubic import BicubicDownSample
from utils.save_utils import save_gen_image, save_latents


class Embedding(nn.Module):
    """
    Module for image embedding
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.encoder = FSencoder.get_trainer(self.opts.device)
        self.e4e, _ = setup_model('pretrained_models/encoder4editing/e4e_ffhq_encode.pt', self.opts.device)

        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)

    def setup_dataloader(self, images: dict[torch.Tensor, list[str]] | list[torch.Tensor], batch_size=None):
        self.dataset = ImagesDataset(images)
        self.dataloader = DataLoader(self.dataset, collate_fn=image_collate, shuffle=False,
                                     batch_size=batch_size or self.opts.batch_size)

    @torch.inference_mode()
    def get_e4e_embed(self, images: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.opts.device
        self.setup_dataloader(images, batch_size=len(images))

        for image, _ in self.dataloader:
            image = image.to(device)
            latent_W = get_latents(self.e4e, image)
            latent_F, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False,
                                             start_layer=0, end_layer=3)
            return {"F": latent_F, "W": latent_W}

    @torch.inference_mode()
    def embedding_images(self, images_to_name: dict[torch.Tensor, list[str]], **kwargs) -> dict[
        str, dict[str, torch.Tensor]]:
        device = self.opts.device
        self.setup_dataloader(images_to_name)

        name_to_embed = defaultdict(dict)
        for image, names in self.dataloader:
            image = image.to(device)

            im_512 = self.downsample_512(image)
            im_256 = self.downsample_256(image)
            im_256_norm = self.normalize(im_256)

            # E4E
            latent_W = get_latents(self.e4e, im_256_norm)

            # FS encoder
            output = self.encoder.test(img=self.normalize(image), return_latent=True)
            latent = output.pop()  # [bs, 512, 16, 16]
            latent_S = output.pop()  # [bs, 18, 512]

            latent_F, _ = self.net.generator([latent_S], input_is_latent=True, return_latents=False,
                                             start_layer=3, end_layer=3, layer_in=latent)  # [bs, 512, 32, 32]

            # BiSeNet
            masks = torch.cat([get_segmentation(image.unsqueeze(0)) for image in self.to_bisenet(im_512)])

            # Mixing if we change the color or shape
            if len(images_to_name) > 1:
                hair_mask = torch.where(masks == 13, torch.ones_like(masks, device=device),
                                        torch.zeros_like(masks, device=device))
                hair_mask = F.interpolate(hair_mask.float(), size=(32, 32), mode='bicubic')

                latent_F_from_W = self.net.generator([latent_W], input_is_latent=True, return_latents=False,
                                                     start_layer=0, end_layer=3)[0]
                latent_F = latent_F + self.opts.mixing * hair_mask * (latent_F_from_W - latent_F)

            for k, names in enumerate(names):
                for name in names:
                    name_to_embed[name]['W'] = latent_W[k].unsqueeze(0)
                    name_to_embed[name]['F'] = latent_F[k].unsqueeze(0)
                    name_to_embed[name]['S'] = latent_S[k].unsqueeze(0)
                    name_to_embed[name]['mask'] = masks[k].unsqueeze(0)
                    name_to_embed[name]['image_256'] = im_256[k].unsqueeze(0)
                    name_to_embed[name]['image_norm_256'] = im_256_norm[k].unsqueeze(0)

            if self.opts.save_all:
                gen_W_im, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False)
                gen_FS_im, _ = self.net.generator([latent_S], input_is_latent=True, return_latents=False,
                                                  start_layer=4, end_layer=8, layer_in=latent_F)

                exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
                output_dir = self.opts.save_all_dir / exp_name
                for name, im_W, lat_W in zip(names, gen_W_im, latent_W):
                    save_gen_image(output_dir, 'W+', f'{name}.png', im_W)
                    save_latents(output_dir, 'W+', f'{name}.npz', latent_W=lat_W)

                for name, im_F, lat_S, lat_F in zip(names, gen_FS_im, latent_S, latent_F):
                    save_gen_image(output_dir, 'FS', f'{name}.png', im_F)
                    save_latents(output_dir, 'FS', f'{name}.npz', latent_S=lat_S, latent_F=lat_F)

        return name_to_embed
