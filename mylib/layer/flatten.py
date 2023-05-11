import numpy as np
from mylib.layer.base_layer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, output_dim: int, input_dim: int = None):
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, X_input):
        return np.reshape(X_input, [-1, np.prod(X_input.shape[1:])])

    def backward(self, dE_dH):

        return None

    def compile(self, input_dim):
        
        return output_dim  # next input_dim
