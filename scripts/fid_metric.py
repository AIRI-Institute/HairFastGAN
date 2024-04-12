import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Encoders import ClipModel
from utils.seed import set_seed
from utils.train import parallel_load_images
from utils.image_utils import list_image_files


def name_path(pair):
    name, path = pair.split(',')
    return name, Path(path)


@torch.inference_mode()
def compute_fid_datasets(datasets, target='celeba', device=torch.device('cuda'), CLIP=False, seed=3407):
    set_seed(seed)
    result = {}

    if CLIP:
        fid = FrechetInceptionDistance(feature=ClipModel(), reset_real_features=False, normalize=False)
    else:
        fid = FrechetInceptionDistance(reset_real_features=False, normalize=False)
    fid.to(device).eval()

    real_dataloader = DataLoader(TensorDataset(datasets[target]), batch_size=128)
    for batch in tqdm(real_dataloader):
        batch = batch[0].to(device)
        fid.update(batch, real=True)

    for key, tensor in datasets.items():
        if key == target:
            continue
        fid.reset()

        fake_dataloader = DataLoader(TensorDataset(tensor), batch_size=128)
        for batch in tqdm(fake_dataloader):
            batch = batch[0].to(device)
            fid.update(batch, real=False)
        result[key] = fid.compute().item()
    return result


def main(args):
    datasets = {}

    source = args.source_dataset.name
    datasets[source] = parallel_load_images(args.source_dataset, list_image_files(args.source_dataset))

    for method, path_dataset in args.methods_dataset:
        datasets[method] = parallel_load_images(path_dataset, list_image_files(path_dataset))

    FIDs = compute_fid_datasets(datasets, target=source, CLIP=False)
    df_fid = pd.DataFrame.from_dict(FIDs, orient='index', columns=['FID'])

    FIDs_CLIP = compute_fid_datasets(datasets, target=source, CLIP=True)
    df_clip = pd.DataFrame.from_dict(FIDs_CLIP, orient='index', columns=['FID_CLIP'])

    df_result = pd.concat([df_fid, df_clip], axis=1).round(2)
    print(df_result)

    os.makedirs(args.output.parent, exist_ok=True)
    df_result.to_csv(args.output, index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics')
    parser.add_argument('--source_dataset', type=Path, help='Dataset with real faces')
    parser.add_argument('--methods_dataset', type=name_path, nargs='+', help='Datasets after applying the method')
    parser.add_argument('--output', type=Path, default='logs/metric.csv', help='Folder for saving logs')
    args = parser.parse_args()

    main(args)
