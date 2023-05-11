import numpy as np
from mylib.layer.base_layer import BaseLayer
from mylib.functions.activation import relu, leaky_relu, sigmoid, softmax, tanh, linear


class Layer(BaseLayer):
    def __init__(self, output_dim: int, activation: str, input_dim: int = None):
        func = {
            'linear': {'self': linear.self, 'diff': linear.diff},
            'relu': {'self': relu.self, 'diff': relu.diff},
            'leaky_relu': {'self': leaky_relu.self, 'diff': leaky_relu.diff},
            'sigmoid': {'self': sigmoid.self, 'diff': sigmoid.diff},
            'softmax': {'self': softmax.self, 'diff': softmax.diff},
            'tanh': {'self': tanh.self, 'diff': tanh.diff},
        }
        self.output_dim = output_dim              # activation = f1(T) -- функция активации
        self.activation: dict = func[activation]
        self.input_dim = input_dim
        self.H0 = None                            # H0 @ W1 + b1 = T1
        self.T = None                             # f1(T1) = H1
        self.W = None
        self.b = None

    def forward(self, X_input):
        self.H0 = X_input
        self.T = self.H0 @ self.W + self.b
        H = self.activation['self'](self.T)
        return H

    def backward(self, dE_dH):
        dh_dt = self.activation['diff'](np.mean(self.T, axis=0, keepdims=True))
        dE_dt = dE_dH * dh_dt

        dE_db = dE_dt

        meanH0 = np.mean(self.H0, axis=0, keepdims=True)
        dE_dW = (meanH0.T @ dE_dt)

        dE_dH0 = dE_dt @ dE_dW.T

        return dE_dH0, dE_dW, dE_db

    def compile(self, input_dim):
        output_dim = self.output_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.ones(output_dim, dtype=float)[np.newaxis, :]
        return output_dim  # next input_dim
