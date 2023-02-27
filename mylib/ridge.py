import numpy as np
from scipy.misc import derivative
from mylib.metrics import Metrics

class Ridge:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, alpha: int, eps: float = 0.01):
        self.X: np.ndarray = np.c_[np.ones(X_train.shape[0]), X_train]
        self.y: np.ndarray = y_train
        self.l, self.d = self.X.shape
        self.alpha: float = alpha
        self.w: np.ndarray = np.zeros((self.d,))
        self.eps = eps
    
    def fit(self):
        self.alg_method()
        # self.gradient_descent()
    
    def predict(self, X) -> np.ndarray:
        return np.dot(X, self.w)
    
    def get_w(self) -> np.ndarray:
        return self.w

    def alg_method(self) -> None:
        self.w = np.linalg.inv(self.X.T.dot(self.X) + self.alpha * np.eye(self.d)).dot(self.X.T).dot(self.y)
    
    # ! Дальше бога нет
    
    # def loss_function(self, w: np.ndarray) -> float:
    #     return (
    #         1/self.l * (
    #             np.sum((self.y - self.predict(self.X, w))**2)
    #             + self.alpha * np.sum(w**2)
    #         )
    #     )
    
    def temp_res(self, w):
        return None
        def score(y_test, y_pred):
            def MSE() -> float:
                return 1/len(y_pred) * np.sum((y_test-y_pred)**2)
            
            return 1 - MSE() * len(y_pred) / np.sum((y_test- np.mean(y_test))**2)
        
        return score(self.y_test, np.dot(self.X_test, w))
    
    def gradient_descent(self) -> None:
        """Метод градиентного спуска"""
        
        def grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.array([-2/self.l*( sum(X[j, i]*(y[j] - np.dot(w, X[j])) for j in range(self.l)) - self.alpha*w[i] ) for i in range(self.d)])
        
        k = 0.5
        t = 15000
        nu = k / t
        w = self.w
        w = np.array([0.0006092385244053693,
 0.0046225533382514,
 0.00023502487792770354,
 0.00017852134576287705,
 0.0017674952450035745,
 3.749354575230174e-05,
 0.010879119419118774,
 0.035778443300405326,
 0.0006056402239344057,
 0.001985168044996785,
 0.0003475593979131137,
 0.00673558706528759,
 0.0002942681389635051])
        w_next = w - nu * grad(w, self.X, self.y)
        while t < 15000: # (buf := np.sum((w_next - w)**2)) >= self.eps and
            if t % 10 == 0:
                print(f'Epoch {t}:\tR^2: {self.temp_res(w_next)}')  # \t{buf}
            nu = k / t
            w_next = w - nu * grad(w, self.X, self.y)
            t += 1
            self.w = w_next  # чтобы были коэфы, если закончить досрочно
        
        self.w = w_next


# def grad1(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
#             return np.array([-2/2*( sum( X[j, i]*(y[j] - np.sum(w*X[j])) for j in range(2)) - 0*w[i] ) for i in range(2)])

# A = np.array(
#     [
#         [1, 2],
#         [3, 4]
#     ]
# )
# w = np.array([10, 5])
# y = np.array([7, 8])

# print(grad1(w, A, y))
