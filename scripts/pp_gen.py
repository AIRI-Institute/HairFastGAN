import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hair_swap import get_parser, HairFast
from scripts.pp_train import Trainer
from utils.train import seed_everything
from utils.image_utils import list_image_files


class ImageException(Exception):
    def __init__(self, image, message="Return image before PP"):
        self.image = image
        self.message = message
        super().__init__(self.message)


def hairfast_wo_pp(hair_fast):
    class RaiseDownsample(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, image):
            image = ((image[0] + 1) / 2).clip(0, 1)
            raise ImageException(image)

    def blend_images(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except ImageException as e:
                return e.image

        return wrapper

    hair_fast.blend.downsample_256 = RaiseDownsample()
    hair_fast.blend.blend_images = blend_images(hair_fast.blend.blend_images)


def load_image(path):
    return T.functional.to_tensor(Image.open(path))


@torch.no_grad()
def load_dataset_images(imgs, dataset_path):
    net_trainer = Trainer()
    source_images = Parallel(n_jobs=-1)(delayed(load_image)(os.path.join(args.FFHQ, img[0])) for img in tqdm(imgs))
    target_images = Parallel(n_jobs=-1)(delayed(load_image)(os.path.join(dataset_path, img[1])) for img in tqdm(imgs))
    tensors_dataloader = DataLoader(list(zip(source_images, target_images)), batch_size=64, pin_memory=False,
                                    shuffle=False, drop_last=False)
    source_files = [os.path.join(args.FFHQ, img[0]) for img in imgs]

    data = []
    total = 0
    for batch in tqdm(tensors_dataloader):
        source, target = [elem.to('cuda') for elem in batch]

        HS_D, _ = net_trainer.generate_mask(source)
        HT_D, HT_E = net_trainer.generate_mask(target)
        target_mask = (1 - HS_D) * (1 - HT_D)

        data.extend(list(zip(
            [source_files[total + i] for i in range(len(source))],
            net_trainer.downsample_256(target).clip(0, 1).cpu(),
            target_mask.cpu(),
            HT_E.cpu(),
        )))
        total += len(source)

    return data


def main(args):
    seed_everything(args.seed)

    # init HairFast
    model_parser = get_parser()
    model_args = model_parser.parse_args([])
    hair_fast = HairFast(model_args)
    hairfast_wo_pp(hair_fast)

    # generate dataset
    os.makedirs(args.output, exist_ok=True)
    images = list_image_files(args.FFHQ)
    face, shape, color = np.array_split(np.random.choice(images, size=3 * args.size), 3)

    exps = []
    for exp in zip(face, shape, color):
        imgs = map(lambda im: im.split('.')[0], exp)
        exps.append([exp[0], f"{'_'.join(imgs)}.png", exp])

    batch = 5_000
    left, right, idx = 0, min(len(exps), batch), 1
    while left < len(exps):
        with tempfile.TemporaryDirectory() as temp_dir:
            for exp in tqdm(exps[left:right]):
                im1, im2, im3 = exp[-1]
                image = hair_fast(args.FFHQ / im1, args.FFHQ / im2, args.FFHQ / im3)
                save_image(image, os.path.join(temp_dir, exp[1]))

            batch_data = load_dataset_images(exps[left:right], temp_dir)
            torch.save(batch_data, os.path.join(args.output, f'pp_part_{idx}.dataset'))
            left = right
            right = min(len(exps), right + batch)
            idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blending dataset')
    parser.add_argument('--FFHQ', type=Path)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--size', type=int, default=10_000)
    parser.add_argument('--output', type=Path, default='input/pp_dataset')
    args = parser.parse_args()

    main(args)
