import numpy as np


def self(x):
    return np.tanh(x)


def diff(x):
    return 1. - np.tanh(x)**2
