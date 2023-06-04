import numpy as np
from math import sqrt


class NaiveBayes:
    def __init__(self):
        self.p_classes: dict = None

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.p_classes = {
            k: {
                'p_class': v/len(y_train),
                'mean': np.mean((tmp := X_train[np.where(np.in1d(y_train, k))]), axis=0),
                'std': np.std(tmp, axis=0),
            } for k, v in zip(*np.unique(y_train, return_counts=True))
        }

        return self

    def predict(self, X_test) -> np.ndarray:
        def f(x, u, o):
            return np.exp(- (x - u)**2 / (2 * o**2)) / (o * sqrt(2*np.pi))

        return np.argmax(tuple(zip(
            np.prod(
                f(np.array(X_test), self.p_classes[_cls]['mean'], self.p_classes[_cls]['std']),
                axis=1
            ) * self.p_classes[_cls]['p_class'] for _cls in sorted(self.p_classes.keys())
            # ! почему [*p_class] не повлияло на результат, даже с [*1] или [*1/p_class]
        )), axis=0)[0]
