import numpy as np
from typing import List
from mylib.layer.layer import Layer
from mylib.optimizer.base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate

    def optimize(self, layers: List[Layer], grad_W, grad_b: np.ndarray):
        for i in range(len(layers)):
            layers[i].W -= self.learning_rate * grad_W[i]
            layers[i].b -= self.learning_rate * grad_b[i]
