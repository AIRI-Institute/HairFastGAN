import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms
from torchvision.utils import save_image

from models.Net import get_segmentation


def equal_replacer(images: list[torch.Tensor]) -> list[torch.Tensor]:
    for i in range(len(images)):
        if images[i].dtype is torch.uint8:
            images[i] = images[i] / 255

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if torch.allclose(images[i], images[j]):
                images[j] = images[i]
    return images


class DilateErosion:
    def __init__(self, dilate_erosion=5, device='cuda'):
        self.dilate_erosion = dilate_erosion
        self.weight = torch.Tensor([
            [False, True, False],
            [True, True, True],
            [False, True, False]
        ]).float()[None, None, ...].to(device)

    def hair_from_mask(self, mask):
        mask = torch.where(mask == 13, torch.ones_like(mask), torch.zeros_like(mask))
        mask = F.interpolate(mask, size=(256, 256), mode='nearest')
        dilate, erosion = self.mask(mask)
        return dilate, erosion

    def mask(self, mask):
        masks = mask.clone().repeat(*([2] + [1] * (len(mask.shape) - 1))).float()
        sum_w = self.weight.sum().item()
        n = len(mask)

        for _ in range(self.dilate_erosion):
            masks = F.conv2d(masks, self.weight,
                             bias=None, stride=1, padding='same', dilation=1, groups=1)
            masks[:n] = (masks[:n] > 0).float()
            masks[n:] = (masks[n:] == sum_w).float()

        hair_mask_dilate, hair_mask_erode = masks[:n], masks[n:]

        return hair_mask_dilate, hair_mask_erode


def poisson_image_blending(final_image, face_image, dilate_erosion=30, maxn=115):
    dilate_erosion = DilateErosion(dilate_erosion=dilate_erosion)
    transform = transforms.ToTensor()

    if isinstance(face_image, str):
        face_image = transform(Image.open(face_image))
    elif not isinstance(face_image, torch.Tensor):
        face_image = transform(face_image)

    final_mask = get_segmentation(final_image.cuda().unsqueeze(0), resize=False)
    face_mask = get_segmentation(face_image.cuda().unsqueeze(0), resize=False)

    hair_target = torch.where(final_mask == 13, torch.ones_like(final_mask),
                              torch.zeros_like(final_mask))
    hair_face = torch.where(face_mask == 13, torch.ones_like(face_mask),
                            torch.zeros_like(face_mask))

    final_mask = F.interpolate(((1 - hair_target) * (1 - hair_face)).float(), size=(1024, 1024), mode='bicubic')
    dilation, _ = dilate_erosion.mask(1 - final_mask)
    mask_save = 1 - dilation[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        final_image_path = os.path.join(temp_dir, 'final_image.png')
        face_image_path = os.path.join(temp_dir, 'face_image.png')
        mask_path = os.path.join(temp_dir, 'mask_save.png')
        save_image(final_image, final_image_path)
        save_image(face_image, face_image_path)
        save_image(mask_save, mask_path)

        out_image_path = os.path.join(temp_dir, 'out_image_path.png')
        result = subprocess.run(
            ["fpie", "-s", face_image_path, "-m", mask_path, "-t", final_image_path, "-o", out_image_path, "-n",
             str(maxn), "-b", "taichi-gpu", "-g", "max"],
            check=True
        )

        return Image.open(out_image_path), Image.open(mask_path)


def list_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []

    for entry in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, entry)
        if os.path.isfile(file_path):
            file_extension = Path(file_path).suffix.lower()
            if file_extension in image_extensions:
                image_files.append(entry)

    return image_files
