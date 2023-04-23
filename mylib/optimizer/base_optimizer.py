import numpy as np
from typing import List
from mylib.layer.layer import Layer


class BaseOptimizer:
    def optimize(self, layers: List[Layer], grad_W: np.ndarray, grad_b: np.ndarray):
        raise NotImplementedError('Не реализован оптимизатор')
