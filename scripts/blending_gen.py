import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train import seed_everything
from utils.image_utils import list_image_files
from hair_swap import get_parser, HairFast
from utils.save_utils import save_latents


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


def align_instead_shape(hair_fast):
    def shape_module(func):
        def wrapper(*args, **kwargs):
            if kwargs.get('align_flag', False):
                return hair_fast.align.align_images(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    def align_module(func):
        def wrapper(*args, **kwargs):
            if 'align_flag' in kwargs:
                kwargs = kwargs.copy()
                kwargs.pop('align_flag')
            return func(*args, **kwargs)

        return wrapper

    hair_fast.align.shape_module = shape_module(hair_fast.align.shape_module)
    hair_fast.align.align_images = align_module(hair_fast.align.align_images)


def main(args):
    seed_everything(args.seed)

    # init HairFast
    model_parser = get_parser()
    model_args = model_parser.parse_args([])
    hair_fast = HairFast(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)

    # generate dataset
    images = list_image_files(args.FFHQ)
    face, shape, color = np.array_split(np.random.choice(images, size=3 * args.size), 3)

    os.makedirs(args.output, exist_ok=True)
    with open(args.output / 'dataset.exps', 'w') as f_exps:
        for imgs in tqdm(zip(face, shape, color)):
            im1, im2, im3 = map(lambda im: im.split('.')[0], imgs)
            print(im1, im2, im3, file=f_exps, flush=True)

            pt1, pt2, pt3 = map(lambda im: args.FFHQ / im, imgs)
            align_shape, align_color, name_to_embed = hair_fast(pt1, pt2, pt3, align_flag=True)
            save_latents(args.output, 'FS', f'{im1}.npz', latent_in=name_to_embed['face']['S'])
            save_latents(args.output, 'FS', f'{im2}.npz', latent_in=name_to_embed['shape']['S'])
            save_latents(args.output, 'FS', f'{im3}.npz', latent_in=name_to_embed['color']['S'])
            save_latents(args.output, 'Align', f'{im1}_{im2}.npz', latent_F=align_shape['latent_F_align'])
            save_latents(args.output, 'Align', f'{im1}_{im3}.npz', latent_F=align_color['latent_F_align'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blending dataset')
    parser.add_argument('--FFHQ', type=Path)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--size', type=int, default=3_000)
    parser.add_argument('--output', type=Path, default='input/blending_dataset')
    args = parser.parse_args()

    main(args)
