import numpy as np
from numpy import ndarray, mean, sum
from math import sqrt


class Metrics():
    def __init__(self, y_test: ndarray, y_pred: ndarray):
        self.y_test = y_test,
        self.y_pred = y_pred

    def __call__(self) -> None:
        print(
            f'MAE:\t{self.MAE()}',
            f'MSE:\t{self.MSE()}',
            f'RMSE:\t{self.RMSE()}',
            f'MAPE:\t{self.MAPE()}',
            f'R^2:\t{self.score()}',
            sep='\n'
        )

    def MAE(self) -> float:
        res = 1/len(self.y_pred) * sum(abs(self.y_test-self.y_pred))
        return res

    def MSE(self) -> float:
        res = 1/len(self.y_pred) * sum((self.y_test-self.y_pred)**2)
        return res

    def RMSE(self) -> float:
        res = sqrt(self.MSE())
        return res

    def MAPE(self) -> float:
        res = 1/len(self.y_pred) * sum(abs((self.y_test-self.y_pred)/self.y_test))
        return res

    def score(self):
        res = 1 - self.MSE() * len(self.y_pred) / sum((self.y_test- mean(self.y_test))**2)
        return res


# y_true = np.array([3, -0.5, 2, 7])
# y_pred = np.array([2.5, 0.0, 2, 8])
# Metrics(y_true, y_pred)()
# # 0.3273...
