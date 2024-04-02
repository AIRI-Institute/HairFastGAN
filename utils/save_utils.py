import os

import numpy as np
import torchvision.transforms as T
from PIL import Image

from models.CtrlHair.util.mask_color_util import mask_to_rgb

toPIL = T.ToPILImage()


def save_gen_image(output_dir, path, name, gen_im):
    if len(gen_im.shape) == 4:
        gen_im = gen_im[0]
    save_im = toPIL(((gen_im + 1) / 2).detach().cpu().clamp(0, 1))

    save_dir = output_dir / path
    os.makedirs(save_dir, exist_ok=True)

    image_path = save_dir / name
    save_im.save(image_path)


def save_vis_mask(output_dir, path, name, mask):
    out_dir = output_dir / path
    os.makedirs(out_dir, exist_ok=True)
    out_mask_path = out_dir / name

    rgb_img = Image.fromarray(mask_to_rgb(mask.detach().cpu().squeeze(), 0))
    rgb_img.save(out_mask_path)


def save_latents(output_dir, path, file_name, **latents):
    save_dir = output_dir / path
    os.makedirs(save_dir, exist_ok=True)

    latent_path = save_dir / file_name
    np.savez(latent_path, **{key: latent.detach().cpu().numpy() for key, latent in latents.items()})
