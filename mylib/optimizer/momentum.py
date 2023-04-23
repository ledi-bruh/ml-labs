import numpy as np
from typing import List
from mylib.layer.layer import Layer
from mylib.optimizer.base_optimizer import BaseOptimizer


class Momentum(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9):
        """
        beta in [0, 1]
        """
        self.learning_rate = learning_rate
        self.beta = beta

    def optimize(self, layers: List[Layer], grad_W: np.ndarray, grad_b: np.ndarray):
        v_W = [np.zeros_like(layer.W) for layer in layers]
        v_b = [np.zeros_like(layer.b) for layer in layers]

        for i in range(len(layers)):
            v_W[i] = self.beta * v_W[i] - self.learning_rate * grad_W[i]
            v_b[i] = self.beta * v_b[i] - self.learning_rate * grad_b[i]

            layers[i].W += v_W[i]
            layers[i].b += v_b[i]
