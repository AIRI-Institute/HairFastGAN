import os
import shutil
import torch
import torchvision.transforms as tform
from PIL import Image


def cpu(xs):
    if not isinstance(xs, (list, tuple)):
        return xs.cpu()
    else:
        return [x.cpu() for x in xs]


def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]
    else:
        return xs


def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])  # -1去掉换行符
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_path)
    return ckpt


def save_checkpoint(obj, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(obj, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def get_img_from_file(file_name, target_device, transform=False):
    img = tform.ToTensor()(Image.open(file_name))  # [0, 1.0] tensor
    if transform:
        val_test_img_transform = tform.Compose([
            # crop face area 190 * 178
            # tform.Lambda(lambda x: x[:, 14:204, :]),
            # center crop and resize on PLI image
            tform.ToPILImage(),
            tform.CenterCrop(170),  # origin # elegant crop
            # tform.CenterCrop(178),
            tform.Resize(256, Image.BICUBIC),  # elegant crop
            # back to tensor
            tform.ToTensor(),
        ])
        img = val_test_img_transform(img)
    return (img * 2 - 1).to(target_device)
