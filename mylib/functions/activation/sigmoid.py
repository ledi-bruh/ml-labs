from scipy.special import expit


def self(x):
    return expit(x)


def diff(x):
    f = expit(x)
    return f * (1. - f)
