import functools
import sys
import time

import numpy as np
import torch


def get_time():
    torch.cuda.current_stream().synchronize()
    return time.time()


def bench_session(func):
    times = []

    @functools.wraps(func)
    def wraps(*args, **kwargs):
        if kwargs.pop('benchmark', False):
            nonlocal times
            start = get_time()

            result = func(*args, **kwargs)

            eval_time = get_time() - start
            times.append(eval_time)

            print(f'\n{len(times)} experiment ended in {eval_time:.3f}(s)', file=sys.stderr)
            print(f'min time: {np.min(times):.3f}(s),'
                  f' median time: {np.median(times):.3f}(s),'
                  f' std time: {np.std(times):.3f}(s)', file=sys.stderr)
            return result
        else:
            return func(*args, **kwargs)

    return wraps
