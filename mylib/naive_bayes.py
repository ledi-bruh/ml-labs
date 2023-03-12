import numpy as np
from math import sqrt


class NaiveBayes:
    mean_std_for_classes: dict
    p_classes: dict
    
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.p_classes = {k: v/len(y_train) for k, v in zip(*np.unique(y_train, return_counts=True))}
        self.mean_std_for_classes = {
            _: {
                'mean': np.mean((tmp := X_train[np.where(np.in1d(y_train, _))]), axis=0),
                'std': np.std(tmp, axis=0)
            } for _ in self.p_classes.keys()
        }
        
        return self
    
    def predict(self, X_test) -> np.ndarray:
        def f(x, u, o):
            return np.exp(- (x - u)**2 / (2 * o**2)) / (o * sqrt(2*np.pi))
        
        return np.argmax(tuple(zip(
            np.prod(
                f(np.array(X_test), self.mean_std_for_classes[_cls]['mean'], self.mean_std_for_classes[_cls]['std']) * self.p_classes[_cls],
                axis=1
            ) for _cls in sorted(self.mean_std_for_classes.keys())
        )), axis=0)[0]
