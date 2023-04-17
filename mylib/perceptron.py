import numpy as np
import numdifftools as ndt
from math import sqrt
from typing import List


class Layer:
    def __init__(self, output_dim: int, activation: callable, input_dim: int = None):
        self.output_dim = output_dim
        self.activation = activation
        self.input_dim = input_dim
        self.W = None
        self.X = None
        self.X_past = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        # self.W = self.W / self.W  # !
        self.X_past = X
        self.X = self.activation(X @ self.W)
        return self.X

    def backward(self, loss: callable, y_true: np.ndarray, bw: np.ndarray = None, alpha: float = 0.01) -> np.ndarray:
        if bw is None:
            bw = ndt.Gradient(loss)(self.X, y_true)#.reshape(-1, 1)
        print(self.X_past.shape, self.W.shape)
        act_diff = ndt.Derivative(self.activation)(self.X_past @ self.W)
        adamar = act_diff * bw
        grad = self.X_past.T @ adamar
        self.W -= alpha * grad
        print('!')
        print(adamar.shape, self.W.shape)
        return adamar @ self.W.T


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
            layer.W = np.vstack((np.random.randn(input_dim, output_dim), np.ones(output_dim)))
            input_dim = layer.output_dim
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1):
        for _ in range(epochs):
            print(f'epoch {_ + 1}/{epochs}')
            y_pred = self.forward(X, self.layers)
            self.backward(y, y_pred, self.layers)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, self.layers)

    def forward(self, X: np.ndarray, layers: List[Layer]) -> np.ndarray:
        cur_X = X
        for layer in layers:
            cur_X = np.c_[cur_X, np.ones((cur_X.shape[0], 1))]
            cur_X = layer.forward(cur_X)
            # print(cur_X)  # !
        print('===== Forward pass ended =====')
        return cur_X

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, layers: List[Layer]):
        print('loss:', self.loss(y_true, y_pred))
        bw = None
        for layer in layers[::-1]:
            bw = layer.backward(self.loss, y_true=y_true, bw=bw)
        print('===== Backward pass ended =====')


X = np.array([
    [1, 2, 3],
    [5, 6, 7],
])

y = np.array([
    2,
    6
])

p = Perceptron([
    Layer(5, lambda X: X, 3),
    Layer(3, lambda X: X),
    Layer(2, lambda X: X),
    Layer(1, lambda X: X),
]).compile(lambda x, y: np.sum((x - y)**2))


# act = lambda X: 2*X + X*X  # 2 + 2x
# grad = ndt.Gradient(act)
# print(ndt.Derivative(act)(X))

# loss = lambda X, Y: np.sum((X - Y)**2)
# print(np.array([[1, 1],[1, 1],[1, 1],[3, 3],[5, 5],[1, 1],]) * ndt.Gradient(loss)([1, 1, 1, 4, 4, 2], [1, 1, 1, 9, 9, 5]).reshape(-1, 1))

p.fit(X, y)
# print(ndt.Gradient(lambda x: x[0]*10 + x[0]*x[1])([3, 4]))
# print(ndt.Derivative(lambda x, y: x*10 + x*y)(3, 4))

print('пук')
