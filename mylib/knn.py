import numpy as np
from math import sqrt


class KnnClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self
    
    def predict(self, X_test) -> np.ndarray:
        return np.array([self.get_type(x_test) for x_test in np.array(X_test)])
    
    def get_type(self, x_test: np.ndarray):
        neighbors = sorted(self.X_train, key=lambda x: self.dist(x_test, x))[:self.n_neighbors]
        
        types = {}
        for neighbor in neighbors:
            type = self.y_train[np.where(np.all(self.X_train == neighbor, axis=1))[0][0]]
            types[type] = types.setdefault(type, 0) + 1
        
        return max(types.items(), key=lambda x: x[1])[0]
    
    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return sqrt(np.sum((a - b)**2))
