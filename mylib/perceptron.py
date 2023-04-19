import numpy as np
from scipy.special import expit
from typing import List, Tuple, Iterable


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def softmax_diff(x):
    f = softmax(x) / x.shape[1]**2
    return f / (1 - f)


class Layer:
    def __init__(self, output_dim: int, activation: str, input_dim: int = None):
        func = {
            'relu': {
                'self': lambda x: np.where(x > 0, x, 0),
                'diff': lambda x: np.where(x > 0, 1, 0),
            },
            'leaky_relu': {
                'self': lambda x: np.where(x > 0, x, 0.01 * x),
                'diff': lambda x: np.where(x > 0, 1, 0.01),
            },
            'sigmoid': {
                'self': lambda x: expit(x),
                'diff': lambda x: expit(x) * (1 - expit(x)),
            },
            'softmax': {
                'self': lambda x: softmax(x),
                'diff': lambda x: softmax_diff(x),
            },
            'tanh': {
                'self': lambda x: np.tanh(x),
                'diff': lambda x: 1. - np.tanh(x)**2,
            },
        }
        self.output_dim = output_dim
        self.activation: dict = func[activation]  # activation = f1(T) -- функция активации
        self.input_dim = input_dim
        self.H0 = None                            # H0 @ W1 + b1 = T1
        self.T = None                             # f1(T1) = H1
        self.W = None
        self.b = None

    def __forward(self, X: np.ndarray) -> np.ndarray:
        self.H0 = X
        self.T = self.H0 @ self.W + self.b
        return self.activation['self'](self.T)      # return H1

    def __backward(self, dE_dh: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        dh_dt = self.activation['diff'](np.mean(self.T, axis=0, keepdims=True))
        dE_dt = dE_dh * dh_dt

        dE_db = dE_dt
        self.b -= alpha * dE_db

        meanH0 = np.mean(self.H0, axis=0, keepdims=True)
        dE_dW = (meanH0.T @ dE_dt)
        self.W -= alpha * dE_dW

        dE_dH0 = dE_dt @ self.W.T
        return dE_dH0


class Perceptron:
    def __init__(self, layers: List[Layer]):
        if layers[0].input_dim is None:
            raise Exception('Не задана входная размерность')
        self.layers = layers
        self.loss = None
        self.func = {
            'mse': {
                'self': lambda true, pred: 0.5 * (pred - true)**2,
                'diff': lambda true, pred: pred - true,
            },
            'sparse_categorical_crossentropy': {
                'self': lambda true, pred: np.sum(-(true * np.where(pred == 0, -1, np.log(pred))), axis=1, keepdims=True),  # ln(0) = 0
                'diff': lambda true, pred: np.sum(-(true / np.where(pred == 0, -1, pred)), axis=1, keepdims=True),
            },
        }

    def compile(self, loss: str):
        self.loss = loss
        input_dim = self.layers[0].input_dim

        for layer in self.layers:
            output_dim = layer.output_dim
            layer.W = np.random.randn(input_dim, output_dim)
            layer.b = np.ones(output_dim, dtype=float)[np.newaxis, :]
            input_dim = output_dim

        return self

    def _batch_split(self, X: np.ndarray, y: np.ndarray, batch_size: int = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if batch_size is None:
            batch_size = X.shape[0]
        for i in range(0, X.shape[0], batch_size):
            yield X[i: i + batch_size], y[i: i + batch_size]

    def _one_hot_y(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        eye = np.eye(n_classes)
        return np.array([eye[y[i]] for i in range(y.shape[0])])

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = None, alpha: float = 1, verbose: bool = False):
        for _ in range(epochs):
            if verbose:
                print(f'epoch {_ + 1}/{epochs}')
            for X_batch, y_batch in self._batch_split(X, y, batch_size=batch_size):
                y_pred = self.__forward(X_batch, self.layers)
                if self.loss == 'sparse_categorical_crossentropy':
                    y_true = self._one_hot_y(y_batch, n_classes=y_pred.shape[1])
                else:
                    y_true = y_batch.reshape(-1, 1)
                self.__backward(y_true=y_true, y_pred=y_pred, layers=self.layers, alpha=alpha, verbose=verbose)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__forward(X, self.layers)

    def __forward(self, X: np.ndarray, layers: List[Layer]) -> np.ndarray:
        pred = X
        for layer in layers:
            pred = layer.__forward(pred)
        return pred

    def __backward(self, y_true: np.ndarray, y_pred: np.ndarray, layers: List[Layer], alpha=0.01, verbose=False):
        if verbose:
            print('loss:', np.sum(self.func[self.loss]['self'](y_true, y_pred)) / y_true.shape[0])
        dE_dh = np.mean(self.func[self.loss]['diff'](y_true, y_pred), axis=0)
        dE_dh = dE_dh[np.newaxis, :]
        for layer in layers[::-1]:
            dE_dh = layer.__backward(dE_dh, alpha=alpha)


# X = np.array([
#     [1, 2, 3],
#     [3, 2, 2],
#     [2, 54, 0.4213],
#     [1, 36.2, 73.4213],
#     [2, -3, 1.4213],
#     [1, 234, 55],
# ])

# y = np.array([
#     0, 2, 1, 0, 1, 0
# ])

# p = Perceptron([
#     Layer(5, 'sigmoid', 3),
#     # Layer(3, lambda X: X),
#     Layer(3, 'softmax'),
# ]).compile(loss='sparse_categorical_crossentropy')


# p.fit(X, y, epochs=1, batch_size=50, verbose=True)
# print(np.argmax(p.predict(np.array([
#     [-2, 2, 5],
#     [0, 2, 0],
#     [1, -5, 8],
#     [1, -7, 0],
# ])), axis=1))
