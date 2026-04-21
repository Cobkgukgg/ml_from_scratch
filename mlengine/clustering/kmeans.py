"""K-Means and K-Means++ clustering."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..utils.math_utils import euclidean_distance


class KMeans:
    """
    K-Means clustering with k-means++ initialization.

    Parameters
    ----------
    n_clusters : int
    init : 'kmeans++' | 'random'
    n_init : int        Number of restarts; best inertia kept
    max_iter : int
    tol : float         Convergence threshold on centroid shift
    random_state : int | None
    """

    def __init__(self, n_clusters: int = 8, init: str = "kmeans++",
                 n_init: int = 10, max_iter: int = 300, tol: float = 1e-4,
                 random_state: int | None = None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centers(self, X: NDArray, rng) -> NDArray:
        n = X.shape[0]
        if self.init == "random":
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            return X[idx].copy()
        # k-means++
        idx0 = rng.integers(n)
        centers = [X[idx0]]
        for _ in range(1, self.n_clusters):
            C = np.array(centers)
            dists = euclidean_distance(X, C).min(axis=1) ** 2
            probs = dists / dists.sum()
            centers.append(X[rng.choice(n, p=probs)])
        return np.array(centers)

    def _fit_once(self, X: NDArray, rng) -> tuple[NDArray, NDArray, float]:
        centers = self._init_centers(X, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.max_iter):
            dists = euclidean_distance(X, centers)
            new_labels = dists.argmin(axis=1)
            new_centers = np.array([
                X[new_labels == k].mean(axis=0) if (new_labels == k).any() else centers[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_centers - centers)
            labels = new_labels
            centers = new_centers
            if shift < self.tol:
                break
        inertia = sum(
            np.sum((X[labels == k] - centers[k]) ** 2)
            for k in range(self.n_clusters)
            if (labels == k).any()
        )
        return centers, labels, float(inertia)

    def fit(self, X: NDArray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        for _ in range(self.n_init):
            centers, labels, inertia = self._fit_once(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
        return self

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        return euclidean_distance(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X: NDArray) -> NDArray:
        return self.fit(X).labels_

    def score(self, X: NDArray) -> float:
        return -self.inertia_


class MiniBatchKMeans(KMeans):
    """Mini-batch variant of K-Means — much faster on large datasets."""

    def __init__(self, n_clusters: int = 8, batch_size: int = 256,
                 max_iter: int = 100, tol: float = 1e-4, random_state: int | None = None):
        super().__init__(n_clusters=n_clusters, n_init=3, max_iter=max_iter,
                         tol=tol, random_state=random_state)
        self.batch_size = batch_size

    def fit(self, X: NDArray) -> "MiniBatchKMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        centers = self._init_centers(X, rng)
        counts = np.ones(self.n_clusters)

        for _ in range(self.max_iter):
            idx = rng.choice(X.shape[0], size=self.batch_size, replace=False)
            Xb = X[idx]
            dists = euclidean_distance(Xb, centers)
            labels = dists.argmin(axis=1)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    counts[k] += mask.sum()
                    lr = 1.0 / counts[k]
                    centers[k] = (1 - lr) * centers[k] + lr * Xb[mask].mean(axis=0)

        self.cluster_centers_ = centers
        self.labels_ = euclidean_distance(X, centers).argmin(axis=1)
        self.inertia_ = float(sum(
            np.sum((X[self.labels_ == k] - centers[k]) ** 2)
            for k in range(self.n_clusters) if (self.labels_ == k).any()
        ))
        return self
