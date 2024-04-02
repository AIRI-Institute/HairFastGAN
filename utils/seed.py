import functools
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def seed_setter(func):
    default_seed = 3407

    @functools.wraps(func)
    def wraps(*args, **kwargs):
        seed = kwargs.pop('seed', None)
        if seed is None:
            seed = default_seed
        set_seed(seed)

        result = func(*args, **kwargs)
        return result

    return wraps
