import numpy as np
from torch import Tensor

from functools import wraps


def get_dim(x: Tensor):
    return len(x.shape) - 2


def to_dim(x: Tensor, dim, channels):
    batch = x.shape[0]
    if dim == 0:
        return x.view(batch, -1)
    x = x.view(batch, -1)
    s = x.shape[1]
    ss = np.round(s ** (1 / dim))
    ks = []
    for i in range(dim):
        ks.append(ss)
    return x.view(batch, channels, *ks)


def require_shape(dim, channels=1):
    """

    :param dim: (-1) tensor need not to be reshaped. 0 for (-1, c), 1 for (-1, c, n), 2 for (-1, c, s, s), etc.
    :param channels:
    :return:
    """

    def temp_shape(f):

        @wraps(f)
        def wrapper(self, *inputs):
            shaped_inputs = []
            origin = -1
            for i, x in enumerate(inputs):
                if origin < 0:
                    origin = get_dim(x)
                shaped_inputs.append(to_dim(x, dim, channels))
            out = f(self, *shaped_inputs)
            return to_dim(out, origin, channels)

        return wrapper

    return temp_shape
