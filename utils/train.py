import os
import pickle
import random
import shutil
import typing as tp

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

from models.Encoders import ClipModel


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class WandbLogger:
    def __init__(self, name='base-name', project='HairFast'):
        self.name = name
        self.project = project

    def start_logging(self):
        wandb.login(key=os.environ['WANDB_KEY'].strip(), relogin=True)
        wandb.init(
            project=self.project,
            name=self.name
        )
        self.wandb = wandb
        self.run_dir = self.wandb.run.dir
        self.train_step = 0

    def log(self, scalar_name: str, scalar: tp.Any):
        self.wandb.log({scalar_name: scalar}, step=self.train_step, commit=False)

    def log_scalars(self, scalars: dict):
        self.wandb.log(scalars, step=self.train_step, commit=False)

    def next_step(self):
        self.train_step += 1

    def save(self, file_path, save_online=True):
        file = os.path.basename(file_path)
        new_path = os.path.join(self.run_dir, file)
        shutil.copy2(file_path, new_path)
        if save_online:
            self.wandb.save(new_path)

    def __del__(self):
        self.wandb.finish()


def toggle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        module = module.replace('torch_utils', 'models.stylegan2.torch_utils')
        module = module.replace('dnnlib', 'models.stylegan2.dnnlib')
        return super().find_class(module, name)


def seed_everything(seed: int = 1729) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_images_to_torch(paths, imgs=None, use_tqdm=True):
    transform = T.PILToTensor()
    tensor = []
    for path in paths:
        if imgs is None:
            pbar = sorted(os.listdir(path))
        else:
            pbar = imgs

        if use_tqdm:
            pbar = tqdm(pbar)

        for img_name in pbar:
            if '.jpg' in img_name or '.png' in img_name:
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).resize((299, 299), resample=Image.LANCZOS)
                tensor.append(transform(img))
    try:
        return torch.stack(tensor)
    except:
        print(paths, imgs)
        return torch.tensor([], dtype=torch.uint8)


def parallel_load_images(paths, imgs):
    assert imgs is not None
    if not isinstance(paths, list):
        paths = [paths]

    list_torch_images = Parallel(n_jobs=-1)(delayed(load_images_to_torch)(
        paths, [i], use_tqdm=False
    ) for i in tqdm(imgs))
    return torch.cat(list_torch_images)


def get_fid_calc(instance='fid.pkl', dataset_path='', device=torch.device('cuda')):
    if os.path.isfile(instance):
        with open(instance, 'rb') as f:
            fid = pickle.load(f)
    else:
        fid = FrechetInceptionDistance(feature=ClipModel(), reset_real_features=False, normalize=True)
        fid.to(device).eval()

        imgs_file = []
        for file in os.listdir(dataset_path):
            if 'flip' not in file and os.path.splitext(file)[1] in ['.png', '.jpg']:
                imgs_file.append(file)

        tensor_images = parallel_load_images([dataset_path], imgs_file).float().div(255)
        real_dataloader = DataLoader(TensorDataset(tensor_images), batch_size=128)
        with torch.inference_mode():
            for batch in tqdm(real_dataloader):
                batch = batch[0].to(device)
                fid.update(batch, real=True)

        with open(instance, 'wb') as f:
            pickle.dump(fid.cpu(), f)
    fid.to(device).eval()

    @torch.inference_mode()
    def compute_fid_datasets(images):
        nonlocal fid, device
        fid.reset()

        fake_dataloader = DataLoader(TensorDataset(images), batch_size=128)
        for batch in tqdm(fake_dataloader):
            batch = batch[0].to(device)
            fid.update(batch, real=False)

        return fid.compute()

    return compute_fid_datasets
