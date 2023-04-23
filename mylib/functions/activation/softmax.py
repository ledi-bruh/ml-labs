import numpy as np


def self(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def diff(x):
    f = self(x) / x.shape[1]**2
    return f / (1 - f)
