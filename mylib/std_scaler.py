import numpy as np


class StdScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.std = np.std(X, axis=0)
        self.mean = np.mean(X, axis=0)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
