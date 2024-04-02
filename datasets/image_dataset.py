import torch
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, images: dict[torch.Tensor, list[str]] | list[torch.Tensor]):
        if isinstance(images, list):
            images = dict.fromkeys(images)

        self.images = list(images)
        self.names = list(images.values())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        if image.dtype is torch.uint8:
            image = image / 255

        names = self.names[index]
        return image, names


def image_collate(batch):
    images = torch.stack([item[0] for item in batch])
    names = [item[1] for item in batch]
    return images, names
