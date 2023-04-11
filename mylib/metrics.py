import numpy as np
from math import sqrt


class Metrics():
    def __init__(self, y_test: np.ndarray, y_pred: np.ndarray):
        self.y_test = np.array(y_test),
        self.y_pred = np.array(y_pred)

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
        res = 1/len(self.y_pred) * np.sum(abs(self.y_test-self.y_pred))
        return res

    def MSE(self) -> float:
        res = 1/len(self.y_pred) * np.sum((self.y_test-self.y_pred)**2)
        return res

    def RMSE(self) -> float:
        res = sqrt(self.MSE())
        return res

    def MAPE(self) -> float:
        res = 1/len(self.y_pred) * np.sum(abs((self.y_test-self.y_pred)/self.y_test))
        return res

    def score(self):
        res = 1 - self.MSE() * len(self.y_pred) / np.sum((self.y_test - np.mean(self.y_test))**2)
        return res


# y_true = np.array([3, -0.5, 2, 7])
# y_pred = np.array([2.5, 0.0, 2, 8])
# Metrics(y_true, y_pred)()
# # 0.3273...


class CMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true: np.ndarray = np.array(y_true)
        self.y_pred: np.ndarray = np.array(y_pred)
    
    def __call__(self) -> None:
        res = '\tprecision\trecall\tf1-score\n'
        for y in np.unique(self.y_true):
            mx = self.confusion_matrix(y)
            res += f'{y}\t{f"{self.precision(mx):.2f}":>9}\t{f"{self.recall(mx):.2f}":>6}\t{f"{self.f1(mx):.2f}":>8}\n'
        res += f'accuracy\t\t\t{f"{self.accuracy(mx):.2f}":>8}\n'
        print(res)
        
    
    def confusion_matrix(self, k: object) -> None:
        mx = [[0, 0], [0, 0]]
        
        for i in range(len(self.y_pred)):
            mx[self.y_pred[i] == self.y_true[i]][self.y_pred[i] == k] += 1
        
        return {
            'tp': mx[1][1],
            'tn': mx[1][0],
            'fp': mx[0][1],
            'fn': mx[0][0],
        }
    
    def accuracy(self, conf_mx) -> float:
        return (conf_mx['tp'] + conf_mx['tn']) / sum(conf_mx.values())
    
    def precision(self, conf_mx) -> float:
        return conf_mx['tp'] / (conf_mx['tp'] + conf_mx['fp'])
    
    def recall(self, conf_mx) -> float:
        return conf_mx['tp'] / (conf_mx['tp'] + conf_mx['fn'])
    
    def f1(self, conf_mx) -> float:
        return 2 / (1 / self.precision(conf_mx) + 1 / self.recall(conf_mx))


# a = np.array([1, 0, 1, 0, 1])
# b = np.array([0, 0, 1, 0, 0])
# CMetrics(b, a)()
