import numpy as np
from typing import List
from mylib.layer.layer import Layer
from mylib.optimizer.base_optimizer import BaseOptimizer


class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, eps: float = 0.1**8):
        """
        beta in [0, 1]
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

    def optimize(self, layers: List[Layer], grad_W: np.ndarray, grad_b: np.ndarray):
        decaying_average_W = [np.zeros_like(layer.W) for layer in layers]
        decaying_average_b = [np.zeros_like(layer.b) for layer in layers]

        for i in range(len(layers)):
            decaying_average_W[i] = self.beta * decaying_average_W[i] + (1 - self.beta) * np.square(grad_W[i])
            decaying_average_b[i] = self.beta * decaying_average_b[i] + (1 - self.beta) * np.square(grad_b[i])

            layers[i].W -= self.learning_rate * grad_W[i] / np.sqrt(decaying_average_W[i] + self.eps)
            layers[i].b -= self.learning_rate * grad_b[i] / np.sqrt(decaying_average_b[i] + self.eps)
