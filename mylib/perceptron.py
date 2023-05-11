import numpy as np
from typing import List, Tuple, Iterable
from mylib.layer.base_layer import BaseLayer
from mylib.functions.loss import mse
from mylib.functions.loss import sparse_categorical_crossentropy as sc_ce
from mylib.optimizer.base_optimizer import BaseOptimizer
from mylib.optimizer.SGD import SGD


class Perceptron:
    def __init__(self, layers: List[BaseLayer]):
        if layers[0].input_dim is None:
            raise Exception('Не задана входная размерность')
        self.layers = layers
        self.optimizer = None
        self.loss = None
        self.func = {
            'mse': {'self': mse.self, 'diff': mse.diff},
            'sparse_categorical_crossentropy': {'self': sc_ce.self, 'diff': sc_ce.diff},
        }

    def compile(self, optimizer: BaseOptimizer = SGD(), loss: str = 'mse'):
        self.optimizer = optimizer
        self.loss = loss
        input_dim = self.layers[0].input_dim

        for layer in self.layers:
            input_dim = layer.compile(input_dim)

        return self

    def _batch_split(self, X: np.ndarray, y: np.ndarray, batch_size: int = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if batch_size is None:
            batch_size = X.shape[0]
        for i in range(0, X.shape[0], batch_size):
            yield X[i: i + batch_size], y[i: i + batch_size]

    def _one_hot_y(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        eye = np.eye(n_classes)
        return np.array([eye[y[i]] for i in range(y.shape[0])])

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = None, verbose: bool = False, return_loss: bool = False):
        losses = []
        for _ in range(epochs):
            if verbose:
                print(f'epoch {_ + 1}/{epochs}')
            loss_epoch = 0
            for X_batch, y_batch in self._batch_split(X, y, batch_size=batch_size):
                y_pred = self.__forward(X_batch, self.layers)
                if self.loss == 'sparse_categorical_crossentropy':
                    y_true = self._one_hot_y(y_batch, n_classes=y_pred.shape[1])
                else:
                    y_true = y_batch.reshape(-1, 1)
                batch_loss = self.__backward(y_true=y_true, y_pred=y_pred, layers=self.layers, verbose=verbose)
                loss_epoch += batch_loss
            losses.append(loss_epoch)

        if return_loss:
            return losses

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__forward(X, self.layers)

    def __forward(self, X: np.ndarray, layers: List[BaseLayer]) -> np.ndarray:
        pred = X
        for layer in layers:
            pred = layer.forward(pred)
        return pred

    def __backward(self, y_true: np.ndarray, y_pred: np.ndarray, layers: List[BaseLayer], verbose=False):
        batch_loss = np.sum(self.func[self.loss]['self'](y_true, y_pred)) / y_true.shape[0]

        if verbose:
            print('loss:', batch_loss)

        dE_dh = np.mean(self.func[self.loss]['diff'](y_true, y_pred), axis=0)
        dE_dH0 = dE_dh[np.newaxis, :]

        grad_W = []
        grad_b = []

        for layer in layers[::-1]:
            dE_dH0, dE_dW, dE_db = layer.backward(dE_dH0)
            grad_b.insert(0, dE_db)
            grad_W.insert(0, dE_dW)
        
        self.optimizer.optimize(self.layers, grad_W, grad_b)
        return batch_loss


# X = np.array([
#     [1, 2, 3],
#     [3, 2, 2],
#     [2, 54, 0.4213],
#     [1, 36.2, 73.4213],
#     [2, -3, 1.4213],
#     [1, 234, 55],
# ])

# y = np.array([
#     0, 2, 1, 0, 1, 0
# ])

# p = Perceptron([
#     Layer(5, 'sigmoid', 3),
#     # Layer(3, lambda X: X),
#     Layer(3, 'softmax'),
# ]).compile(loss='sparse_categorical_crossentropy')


# p.fit(X, y, epochs=1, batch_size=50, verbose=True)
# print(np.argmax(p.predict(np.array([
#     [-2, 2, 5],
#     [0, 2, 0],
#     [1, -5, 8],
#     [1, -7, 0],
# ])), axis=1))
