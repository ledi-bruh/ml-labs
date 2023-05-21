import numpy as np


class KMeans:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, data: np.ndarray, eps=0.001):
        centroids = np.random.default_rng().choice(data, size=self.n_clusters, replace=False, axis=0)
        wcss1 = np.inf
        wcss2 = 0
        labels = []

        while np.abs(wcss1 - wcss2) > eps:
            wcss1 = wcss2

            res = [[] for _ in range(self.n_clusters)]
            labels = []

            for point in data:
                idx = np.argmin([self.dist(point, c) for c in centroids])
                res[idx].append(point)
                labels.append(idx)

            wcss2 = 0
            for i in range(self.n_clusters):
                centroids[i] = np.mean(res[i], axis=0)
                wcss2 += np.sum([self.dist(point_in_c, centroids[i])**2 for point_in_c in res[i]])
        
        self.labels_ = labels
        self.cluster_centers_ = centroids

        return self

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.sqrt(np.sum((a - b)**2))
