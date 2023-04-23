import numpy as np
from typing import List
from mylib.layer.layer import Layer
from mylib.optimizer.base_optimizer import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.99, eps: float = 0.1**8):
        """
        beta in [0, 1]
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def optimize(self, layers: List[Layer], grad_W: np.ndarray, grad_b: np.ndarray):
        v_W = [np.zeros_like(layer.W) for layer in layers]
        v_b = [np.zeros_like(layer.b) for layer in layers]
        
        norm_v_W = v_W.copy()
        norm_v_b = v_b.copy()
        
        decaying_average_W = [np.zeros_like(layer.W) for layer in layers]
        decaying_average_b = [np.zeros_like(layer.b) for layer in layers]
        
        norm_decaying_average_W = decaying_average_W.copy()
        norm_decaying_average_b = decaying_average_b.copy()

        for i in range(len(layers)):
            v_W[i] = self.beta1 * v_W[i] + (1 - self.beta1) * grad_W[i]
            v_b[i] = self.beta1 * v_b[i] + (1 - self.beta1) * grad_b[i]
            
            decaying_average_W[i] = self.beta2 * decaying_average_W[i] + (1 - self.beta2) * np.square(grad_W[i])
            decaying_average_b[i] = self.beta2 * decaying_average_b[i] + (1 - self.beta2) * np.square(grad_b[i])
            
            norm_v_W[i] = v_W[i] / (1 - self.beta1**(i + 1))
            norm_v_b[i] = v_b[i] / (1 - self.beta1**(i + 1))
            
            norm_decaying_average_W[i] = decaying_average_W[i] / (1 - self.beta2**(i + 1))
            norm_decaying_average_b[i] = decaying_average_b[i] / (1 - self.beta2**(i + 1))

            layers[i].W -= self.learning_rate * norm_v_W[i] / (np.sqrt(norm_decaying_average_W[i]) + self.eps)
            layers[i].b -= self.learning_rate * norm_v_b[i] / (np.sqrt(norm_decaying_average_b[i]) + self.eps)
