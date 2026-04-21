"""Missing value imputation strategies."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class SimpleImputer:
    """
    Fill missing values with mean, median, most_frequent, or constant.

    Parameters
    ----------
    strategy : 'mean' | 'median' | 'most_frequent' | 'constant'
    fill_value : scalar   Used when strategy='constant'
    missing_values : scalar   Value treated as missing (default np.nan)
    """

    def __init__(self, strategy: str = "mean", fill_value=0,
                 missing_values=np.nan):
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values

    def _is_missing(self, X: NDArray) -> NDArray:
        if self.missing_values is np.nan or (isinstance(self.missing_values, float) and np.isnan(self.missing_values)):
            return np.isnan(X)
        return X == self.missing_values

    def fit(self, X: NDArray, y=None) -> "SimpleImputer":
        X = np.asarray(X, dtype=float)
        mask = self._is_missing(X)
        self.statistics_ = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~mask[:, j]]
            if self.strategy == "mean":
                self.statistics_[j] = valid.mean() if len(valid) else 0.0
            elif self.strategy == "median":
                self.statistics_[j] = np.median(valid) if len(valid) else 0.0
            elif self.strategy == "most_frequent":
                if len(valid):
                    vals, counts = np.unique(valid, return_counts=True)
                    self.statistics_[j] = vals[counts.argmax()]
                else:
                    self.statistics_[j] = 0.0
            else:
                self.statistics_[j] = self.fill_value
        return self

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float).copy()
        mask = self._is_missing(X)
        for j in range(X.shape[1]):
            X[mask[:, j], j] = self.statistics_[j]
        return X

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)


class KNNImputer:
    """
    Impute missing values using k-nearest neighbours (uniform weights).
    Distances computed only over observed features between samples.
    """

    def __init__(self, n_neighbors: int = 5, missing_values=np.nan):
        self.n_neighbors = n_neighbors
        self.missing_values = missing_values

    def fit(self, X: NDArray, y=None) -> "KNNImputer":
        self.X_train_ = np.asarray(X, dtype=float).copy()
        return self

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        for i in range(X.shape[0]):
            missing_cols = np.where(nan_mask[i])[0]
            if len(missing_cols) == 0:
                continue
            obs_cols = np.where(~nan_mask[i])[0]
            dists = []
            for j in range(self.X_train_.shape[0]):
                shared = obs_cols[~np.isnan(self.X_train_[j, obs_cols])]
                if len(shared) == 0:
                    dists.append(np.inf)
                    continue
                dists.append(np.sqrt(np.sum((X[i, shared] - self.X_train_[j, shared]) ** 2)))
            dists = np.array(dists)
            nn_idx = np.argsort(dists)[:self.n_neighbors]
            for c in missing_cols:
                vals = self.X_train_[nn_idx, c]
                valid_vals = vals[~np.isnan(vals)]
                X[i, c] = valid_vals.mean() if len(valid_vals) else 0.0
        return X

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)
