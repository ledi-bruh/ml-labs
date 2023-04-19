import numpy as np
import numdifftools as ndt
from typing import List


class Layer:
    def __init__(self, output_dim: int, activation: callable, input_dim: int = None):
        self.output_dim = output_dim
        self.activation = activation        # activation = f1(T) -- функция активации
        self.input_dim = input_dim
        self.H0 = None                      # H0 @ W1 + b1 = T1
        self.T = None                       # f1(T1) = H1
        self.W = None
        self.b = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.H0 = X
        self.T = self.H0 @ self.W + self.b
        return self.activation(self.T)      # return H1

    def backward(self, dE_dh: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        dh_dt = ndt.Derivative(self.activation)(np.mean(self.T, axis=0))[np.newaxis, :]
        dE_dt = dE_dh * dh_dt

        dE_db = dE_dt
        self.b -= alpha * dE_db

        meanH0 = np.mean(self.H0, axis=0)[np.newaxis, :]
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

    def compile(self, loss: callable):
        self.loss = loss
        input_dim = self.layers[0].input_dim

        for layer in self.layers:
            output_dim = layer.output_dim
            layer.W = np.random.randn(input_dim, output_dim, )
            layer.b = np.ones(output_dim, dtype=float)[np.newaxis, :]
            input_dim = output_dim

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, alpha=0.01):
        for _ in range(epochs):
            print(f'epoch {_ + 1}/{epochs}')
            y_pred = self.forward(X, self.layers)
            # ! для классификации что с y делать?
            # TODO: тут преобразовать y
            self.backward(y_true=y, y_pred=y_pred, layers=self.layers, alpha=alpha)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, self.layers)

    def forward(self, X: np.ndarray, layers: List[Layer]) -> np.ndarray:
        pred = X
        for layer in layers:
            pred = layer.forward(pred)
        print('===== Forward pass ended =====')
        return pred

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, layers: List[Layer], alpha=0.01):
        # print('loss:', self.loss(y_true, y_pred))  # ! ? ? ? ? ? ?
        y_true = y_true.reshape(-1, 1)
        dE_dh = np.mean(ndt.Derivative(lambda pred: self.loss(y_true, pred))(y_pred), axis=0)
        dE_dh = dE_dh[np.newaxis, :]
        for layer in layers[::-1]:
            dE_dh = layer.backward(dE_dh, alpha=alpha)
        print('===== Backward pass ended =====')


X = np.array([
    [1, 2, 3],
    [3, 2, 2],
    [1, 0, 1],
    [4, 0, 0],
])

y = np.array([
    1, 1, 0, 1
])

# p = Perceptron([
#     Layer(5, lambda X: np.maximum(0, X), 3),
#     # Layer(3, lambda X: X),
#     Layer(2, lambda X: np.maximum(0, X)),
#     Layer(1, lambda X: np.maximum(0, X)),
# ]).compile(lambda true, pred: (pred - true)**2)


# p.fit(X, y)

# ls = lambda x, y: np.sum((x - y)**2)
# y_true = np.array([[1], [1]])
# y_pred = np.array([[4], [2]])

# x = np.array([[1, 0],
#               [0, 3],
#               [5, 1],
#               [0, 1],])
# y = np.array([
#               [1, 0],
#               [0, 1],
#               [0, 1],
#               [0, 1],
# ])
# E = lambda true, pred: (pred - true)**2
# Q = lambda Y, X: np.sum(E(Y, X)) / Y.shape[0]
# # print(
# #     np.mean(ndt.Derivative(lambda pred: E(y, pred))(x), axis=0)
# # )
# # print(
# #     np.mean(ndt.Derivative(lambda pred: E(y_true, pred))(y_pred), axis=0)
# # )
