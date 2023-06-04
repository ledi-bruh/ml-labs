import numpy as np
from mylib.std_scaler import StdScaler


class PCA:
    def __init__(self, n_components, scaler=None):
        self.n_components = n_components
        self.scaler = StdScaler() if scaler is None else scaler
        self.X = None
        self.T = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.X = self.scaler.fit_transform(X)
        cov_mx = np.cov(self.X.T)
        values, vectors = np.linalg.eig(cov_mx)
        pairs = sorted(zip(values, vectors), reverse=True)[:self.n_components]
        self.T = np.c_[[vector for _, vector in pairs]].T
        return self.X @ self.T
    
    def transform(self, X) -> np.ndarray:
        X = self.scaler.transform(X)
        return X @ self.T


# pca = PCA(6)
# a = np.array([
#     [1, 2, 2, -4, 5, 66, 7, 81, 4, 9],
#     [10, 20, 35, 43, 52, 76, 77, 28, 79, 109],
#     [3, 12, 16, 23, 22, 56, 27, 18, 49, 19]
# ])
# b = pca.fit_transform(a, np.array([1, 10, 3]))
# print(type(b))
