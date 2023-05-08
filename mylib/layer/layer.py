from mylib.functions.activation import relu, leaky_relu, sigmoid, softmax, tanh, linear


class Layer:
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
