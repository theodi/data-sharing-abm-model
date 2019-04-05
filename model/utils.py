import numpy as np


def min_max_scaler(vals, axis=None):
    """
    A simple minmax scaler, which also deals with some specific edge cases
    """
    if axis == 0 or axis == -1:
        max = np.nanmax(vals, axis=axis)
        min = np.nanmin(vals, axis=axis)
        return np.nan_to_num(vals / np.expand_dims(max - min, axis=axis))
    else:
        max = np.max(vals)
        min = np.min(vals)
        if min == max:
            return np.ones_like(vals)
        return (vals - min) / (max - min)


def multinomial(M, rng):
    """
    Given a probability matrix M of dimensions [n_1, n_2, ..., n_m], with the probabilities summing
    up to one over the last axis, this function return a matrix of the same dimension of M where
    every array [n_1, ..., n_{m-1}, :] has exactly one 1 and the rest 0, indicating a choice
    """
    cumulative = np.cumsum(M, axis=-1)
    rands = rng.uniform(size=M.shape[:-1])
    smaller = (np.expand_dims(rands, axis=-1) < cumulative).astype(int)
    choice = smaller.cumsum(axis=-1) == 1
    return choice
