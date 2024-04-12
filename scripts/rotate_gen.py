import argparse
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.rotate_train import Trainer
from utils.train import seed_everything
from utils.image_utils import list_image_files

toTensor = T.ToTensor()
net_trainer = Trainer()


def load_image(path):
    return toTensor(Image.open(path))


@torch.no_grad()
def load_dataset_images(imgs):
    tensors_images = Parallel(n_jobs=-1)(
        delayed(load_image)(os.path.join(args.FFHQ, str(img))) for img in tqdm(imgs))
    tensors_dataloader = DataLoader(tensors_images, batch_size=32, pin_memory=False, shuffle=False, drop_last=False)

    images, key_points, latents = [], [], []
    for batch in tqdm(tensors_dataloader):
        batch = batch.to(net_trainer.device)

        images_256 = net_trainer.downsample_256(batch).clip(0, 1)
        images.extend(images_256.cpu())
        latents.extend(net_trainer.generate_latents(images_256 * 2 - 1).cpu())
        key_points.extend(net_trainer.generate_key_points(batch).cpu())

    return images, key_points, latents


def main(args):
    seed_everything(args.seed)

    images = list_image_files(args.FFHQ)
    random.shuffle(images)

    images, key_points, latents = load_dataset_images(images[:args.size])

    torch.save({'images': images, 'key_points': key_points, 'latents': latents}, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotate dataset')
    parser.add_argument('--FFHQ', type=Path)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--size', type=int, default=10_000)
    parser.add_argument('--output', type=Path, default='input/rotate_dataset.pkl')
    args = parser.parse_args()

    main(args)
