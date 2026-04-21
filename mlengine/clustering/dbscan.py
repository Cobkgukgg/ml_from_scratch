"""DBSCAN and HDBSCAN-lite clustering."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from collections import deque


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps : float         Neighbourhood radius
    min_samples : int   Min points to form a core point
    metric : str        'euclidean' | 'cosine'
    """

    NOISE = -1
    UNVISITED = -2

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def _pairwise_distances(self, X: NDArray) -> NDArray:
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
            Xn = X / norms
            return 1 - Xn @ Xn.T
        # euclidean
        sq = (X ** 2).sum(axis=1)
        return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))

    def fit(self, X: NDArray) -> "DBSCAN":
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        D = self._pairwise_distances(X)
        labels = np.full(n, self.UNVISITED, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != self.UNVISITED:
                continue
            neighbors = np.where(D[i] <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                labels[i] = self.NOISE
                continue
            # Expand cluster
            labels[i] = cluster_id
            queue = deque(neighbors.tolist())
            while queue:
                j = queue.popleft()
                if labels[j] == self.NOISE:
                    labels[j] = cluster_id
                if labels[j] != self.UNVISITED:
                    continue
                labels[j] = cluster_id
                j_neighbors = np.where(D[j] <= self.eps)[0]
                if len(j_neighbors) >= self.min_samples:
                    queue.extend(j_neighbors.tolist())
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(
            np.array([(D[i] <= self.eps).sum() >= self.min_samples for i in range(n)])
        )[0]
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X: NDArray) -> NDArray:
        return self.fit(X).labels_
