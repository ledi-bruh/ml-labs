import numpy as np
from scipy.stats import mode
from typing import Generator


class Bagging:
    def __init__(self, type_task: str, estimator=None, n_estimators: int = 1, kwargs: dict = {}):
        if type_task not in ('R', 'C'):
            raise Exception('Неизвестный тип задачи. [R/C]')
        self.type_task = type_task
        self.estimators = [estimator(**kwargs) for _ in range(n_estimators)]
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray):
        data = np.c_[X, y]
        for _ in range(self.n_estimators):
            model_data = self.__bootstrap(data)
            self.estimators[_].fit(model_data[:, :-1], model_data[:, -1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        def gen(X: np.ndarray, func: callable) -> Generator:
            for x in X:
                yield func([model.predict(x.reshape(1, -1))[0] for model in self.estimators])

        if self.type_task == 'R':
            return list(gen(X, lambda x: np.mean(x)))
        return np.array(list(gen(X, lambda x: mode(x).mode[0])))

    def __bootstrap(self, data: np.ndarray) -> np.ndarray:
        return np.random.default_rng().choice(data, axis=0, size=data.shape[0])
