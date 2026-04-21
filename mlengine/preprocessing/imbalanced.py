"""Techniques for handling class imbalance: SMOTE, RandomOverSampler, RandomUnderSampler."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class RandomOverSampler:
    """Randomly duplicate minority class samples."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit_resample(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_res, y_res = [X], [y]
        for c, n in zip(classes, counts):
            if n < max_count:
                idx = np.where(y == c)[0]
                extra = rng.choice(idx, size=max_count - n, replace=True)
                X_res.append(X[extra])
                y_res.append(y[extra])
        X_out = np.vstack(X_res)
        y_out = np.concatenate(y_res)
        perm = rng.permutation(len(y_out))
        return X_out[perm], y_out[perm]


class RandomUnderSampler:
    """Randomly remove majority class samples."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit_resample(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        X_res, y_res = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            chosen = rng.choice(idx, size=min_count, replace=False)
            X_res.append(X[chosen])
            y_res.append(y[chosen])
        X_out = np.vstack(X_res)
        y_out = np.concatenate(y_res)
        perm = rng.permutation(len(y_out))
        return X_out[perm], y_out[perm]


class SMOTE:
    """
    Synthetic Minority Oversampling TEchnique (Chawla et al., 2002).
    Generates synthetic samples by interpolating between minority class neighbours.

    Parameters
    ----------
    k_neighbors : int   Number of nearest neighbours to interpolate between
    random_state : int | None
    """

    def __init__(self, k_neighbors: int = 5, random_state: int | None = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def _knn(self, X_minority: NDArray, k: int) -> NDArray:
        """Return k nearest neighbour indices for each sample in X_minority."""
        sq = (X_minority ** 2).sum(axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X_minority @ X_minority.T, 0))
        np.fill_diagonal(D, np.inf)
        return np.argsort(D, axis=1)[:, :k]

    def fit_resample(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_res, y_res = [X], [y]

        for c, n in zip(classes, counts):
            if n >= max_count:
                continue
            X_min = X[y == c]
            n_synthetic = max_count - n
            nn_idx = self._knn(X_min, min(self.k_neighbors, len(X_min) - 1))
            synthetic = []
            for _ in range(n_synthetic):
                i = rng.integers(len(X_min))
                nn = nn_idx[i, rng.integers(nn_idx.shape[1])]
                lam = rng.uniform(0, 1)
                synthetic.append(X_min[i] + lam * (X_min[nn] - X_min[i]))
            X_res.append(np.array(synthetic))
            y_res.append(np.full(n_synthetic, c))

        X_out = np.vstack(X_res)
        y_out = np.concatenate(y_res)
        perm = rng.permutation(len(y_out))
        return X_out[perm], y_out[perm]
