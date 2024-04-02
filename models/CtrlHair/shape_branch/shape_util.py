import torch

from models.CtrlHair.global_value_utils import HAIR_IDX


def mask_label_to_one_hot(img):
    # img = img * 255
    img[img == 255] = 19
    bs, _, h, w = img.size()
    nc = 19
    input_label = torch.FloatTensor(bs, nc + 1, h, w).zero_().to(img.device)
    input_semantics = input_label.scatter_(1, img.long(), 1.0)
    input_semantics = input_semantics[:, :-1, :, :]
    return input_semantics


def mask_one_hot_to_label(one_hot):
    mask = torch.argmax(one_hot, dim=1)
    mask[one_hot.max(dim=1)[0] == 0] = 255
    return mask


def split_hair_face(mask):
    hair = mask[:, [HAIR_IDX], :, :]
    face = torch.cat([mask[:, :HAIR_IDX, :, :], mask[:, HAIR_IDX + 1:, :, :]], dim=1)
    return hair, face
