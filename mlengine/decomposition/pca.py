"""Principal Component Analysis via eigendecomposition and SVD."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class PCA:
    """
    Principal Component Analysis.

    Parameters
    ----------
    n_components : int | float | 'mle' | None
        - int: exact number of components
        - float in (0,1): fraction of variance to retain
        - None: keep all
    whiten : bool   Normalise components to unit variance
    """

    def __init__(self, n_components=None, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: NDArray, y=None) -> "PCA":
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.explained_variance_ = s ** 2 / (n - 1)
        total = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total
        self.singular_values_ = s

        k = self._resolve_n_components(p, self.explained_variance_ratio_)
        self.n_components_ = k
        self.components_ = Vt[:k]          # (k, p)
        self.noise_variance_ = (
            self.explained_variance_[k:].mean() if k < p else 0.0
        )
        return self

    def _resolve_n_components(self, p, ratios):
        nc = self.n_components
        if nc is None:
            return p
        if isinstance(nc, float):
            cumsum = np.cumsum(ratios)
            return int(np.searchsorted(cumsum, nc) + 1)
        return min(int(nc), p)

    def transform(self, X: NDArray) -> NDArray:
        Xc = np.asarray(X, dtype=float) - self.mean_
        Xt = Xc @ self.components_.T
        if self.whiten:
            Xt /= (np.sqrt(self.explained_variance_[:self.n_components_]) + 1e-10)
        return Xt

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: NDArray) -> NDArray:
        if self.whiten:
            X_reduced = X_reduced * np.sqrt(self.explained_variance_[:self.n_components_])
        return X_reduced @ self.components_ + self.mean_

    @property
    def cumulative_variance_ratio_(self) -> NDArray:
        return np.cumsum(self.explained_variance_ratio_)
