"""Truncated SVD (LSA), Randomized SVD, and NMF."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class TruncatedSVD:
    """
    Dimensionality reduction via truncated SVD (similar to PCA but no centering).
    Useful for sparse matrices / text data (LSA).
    """

    def __init__(self, n_components: int = 2, n_iter: int = 5, random_state: int | None = None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: NDArray, y=None) -> "TruncatedSVD":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        k = min(self.n_components, min(n, p))
        # Randomized SVD (Halko et al., 2009)
        Omega = rng.standard_normal((p, k + 10))
        Y = X @ Omega
        for _ in range(self.n_iter):
            Y = X @ (X.T @ Y)
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ X
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        self.components_ = Vt[:k]
        self.singular_values_ = s[:k]
        self.explained_variance_ = s[:k] ** 2 / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / (X ** 2).sum() * (n - 1)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return np.asarray(X, dtype=float) @ self.components_.T

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: NDArray) -> NDArray:
        return X_reduced @ self.components_


class NMF:
    """
    Non-negative Matrix Factorisation via multiplicative update rules.
    Factorises X ≈ W @ H with W, H ≥ 0.
    """

    def __init__(self, n_components: int = 2, max_iter: int = 200,
                 tol: float = 1e-4, random_state: int | None = None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        X = np.asarray(X, dtype=float)
        assert (X >= 0).all(), "NMF requires non-negative input."
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        k = self.n_components
        W = rng.uniform(0, 1, (n, k))
        H = rng.uniform(0, 1, (k, p))
        eps = 1e-10
        for _ in range(self.max_iter):
            H *= (W.T @ X) / (W.T @ W @ H + eps)
            W *= (X @ H.T) / (W @ H @ H.T + eps)
        self.components_ = H
        self.reconstruction_err_ = float(np.linalg.norm(X - W @ H, "fro"))
        return W

    def fit(self, X: NDArray, y=None) -> "NMF":
        self.fit_transform(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Project new data onto H (held fixed) via NNLS-like update."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        W = rng.uniform(0, 1, (X.shape[0], self.n_components))
        H = self.components_
        eps = 1e-10
        for _ in range(100):
            W *= (X @ H.T) / (W @ H @ H.T + eps)
        return W
