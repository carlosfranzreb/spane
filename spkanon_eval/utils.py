import random
from gc import collect
import inspect

import numpy as np
import torch
from torch.cuda import empty_cache


def reset(model):
    """Call the model's reset method if exists and empty CUDA cache and garbage."""
    if hasattr(model, "reset"):
        model.reset()
        empty_cache()
        collect()


def seed_everything(seed: int):
    """Set the seed for Python, Numpy and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_creation_tracker(func):
    """
    This function can be used to debug memory leakage. I've used this to find out
    that NeMo's hifigan leaks memory when an OOM occurs. It's used like this:
    `torch.tensor = tensor_creation_tracker(torch.tensor)`
    """

    def wrapper(*args, **kwargs):
        t = func(*args, **kwargs)
        if isinstance(t, torch.Tensor):
            if t.ndim == 3 and t.shape[1] in [64, 128] and t.shape[2] > 500:
                caller_frame = inspect.currentframe().f_back
                filename = caller_frame.f_code.co_filename
                lineno = caller_frame.f_lineno
                print(
                    f"Tracked tensor created at {filename}:{lineno} with shape {t.shape}"
                )
        return t

    if torch.jit.is_scripting():
        return func
    else:
        return wrapper
