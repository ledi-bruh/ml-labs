import numpy as np


def self(x):
    return np.where(x > 0, x, 0)


def diff(x):
    return np.where(x > 0, 1., 0)
