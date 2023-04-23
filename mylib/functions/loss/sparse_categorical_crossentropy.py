import numpy as np


def self(true, pred):
    # ln(0) = 0
    return -np.sum((true * np.where(pred == 0, 0, np.log(pred))), axis=1, keepdims=True)


def diff(true, pred):
    return pred - true
    return -np.sum((true / np.where(pred == 0, 0, pred)), axis=1, keepdims=True)
