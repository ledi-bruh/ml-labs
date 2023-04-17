import numpy as np
from scipy.stats import mode
from typing import Callable


def gini(data: np.ndarray) -> float:
    return 1 - np.sum((np.unique(data, return_counts=True)[1] / data.shape[0])**2)


def mse(data: np.ndarray) -> float:
    return 1/data.shape[0] * np.sum((data - data.mean())**2)


class Node:
    def __init__(self):
        self.left: object = None
        self.right: object = None
        self.score: float  # MSE / Gini
        self.value: dict = {
            'res': None,
            'predictor': None
        }


class CART:
    def __init__(self, type_task: str, max_depth: int = 3, criterion: Callable = None):
        tasks = {
            'R': mse,
            'C': gini
        }
        self.type_task = type_task
        self.root = Node()
        self.max_depth = max_depth
        self.criterion = tasks[type_task] if criterion is None else criterion

    def fit(self, X: np.ndarray, y: np.ndarray):
        data = np.c_[X, y]
        self.root = self.build_tree(data, self.max_depth, self.criterion)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(list(self.__find(x, self.root) for x in np.array(X)))

    def __find(self, x: np.ndarray, node: Node) -> float:
        if None in (node.left, node.right):
            return node.value['res']
        if x[node.value['predictor']['j']] <= node.value['predictor']['t']:
            return self.__find(x, node.left)
        return self.__find(x, node.right)

    def build_tree(self, data: np.ndarray, max_depth: int, criterion: Callable, depth: int = 0) -> Node:
        node = Node()
        node.score = criterion(data[:, -1])
        node.value['res'] = self.__get_stats(data)

        if depth > max_depth or len(data) == 1 or node.score == 0:
            return node

        best = self.__best_split(data, criterion)
        node.value['predictor'] = best['predictor']

        node.left = self.build_tree(best['left'], max_depth, criterion, depth + 1)
        node.right = self.build_tree(best['right'], max_depth, criterion, depth + 1)
        return node

    def __best_split(self, data: np.ndarray, criterion: Callable) -> dict:
        ni, nj = np.array(data).shape
        best = {
            'left': None,
            'right': None,
            'score': np.inf,
            'predictor': {
                't': None,
                'j': None,
            },
        }
        for j in range(nj - 1):  # ?
            for i in range(ni):
                data_left, data_right = self.__split(data[i, j], j, data)
                if 0 in (len(data_left), len(data_right)):
                    continue
                score = self.__check_split(np.array(data_left), np.array(data_right), criterion)
                if score < best['score']:
                    best['score'] = score
                    best['left'] = data_left
                    best['right'] = data_right
                    best['predictor']['t'] = data[i, j]
                    best['predictor']['j'] = j
        return best

    def __split(self, t: float, _j: int, data: np.ndarray) -> tuple:
        return data[data[:, _j] <= t], data[data[:, _j] > t]

    def __check_split(self, l_data: np.ndarray, r_data: np.ndarray, criterion: Callable) -> float:
        li = l_data.shape[0]
        ri = r_data.shape[0]
        return li/(li+ri) * criterion(l_data[:, -1]) + ri/(li+ri) * criterion(r_data[:, -1])

    def __get_stats(self, data: np.ndarray) -> float:
        if self.type_task == 'R':
            return np.mean(data[:, -1])
        elif self.type_task == 'C':
            return mode(data[:, -1]).mode[0]
        raise Exception('Неизвестный тип задачи. [R/C]')
