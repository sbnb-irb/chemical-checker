from __future__ import division
import numpy as np


def calc_p(ncal, ngt, neq, smoothing=False, f=None):
    if smoothing:
        if f is None:
            f = np.random.uniform(0, 1)
        return (ngt + (neq + 1) * f) / (ncal + 1)
    else:
        return (ngt + neq + 1) / (ncal + 1)
