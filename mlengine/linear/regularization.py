"""Regularization utilities: ElasticNet, L1/L2 penalties, early stopping."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def l1_penalty(coef: NDArray) -> float:
    return float(np.sum(np.abs(coef)))


def l2_penalty(coef: NDArray) -> float:
    return float(0.5 * np.sum(coef ** 2))


def elastic_net_penalty(coef: NDArray, l1_ratio: float = 0.5) -> float:
    return l1_ratio * l1_penalty(coef) + (1 - l1_ratio) * l2_penalty(coef)


def l1_gradient(coef: NDArray) -> NDArray:
    return np.sign(coef)


def l2_gradient(coef: NDArray) -> NDArray:
    return coef.copy()


def elastic_net_gradient(coef: NDArray, l1_ratio: float = 0.5) -> NDArray:
    return l1_ratio * l1_gradient(coef) + (1 - l1_ratio) * l2_gradient(coef)


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class ElasticNet:
    """
    ElasticNet regression (L1 + L2) via coordinate descent.
    Combines the sparsity of Lasso with the grouping effect of Ridge.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 n_iter: int = 1000, tol: float = 1e-4, fit_intercept: bool = True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X: NDArray, y: NDArray) -> "ElasticNet":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        if self.fit_intercept:
            self.intercept_ = y.mean()
            y = y - self.intercept_
        else:
            self.intercept_ = 0.0
        coef = np.zeros(p)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        for _ in range(self.n_iter):
            coef_old = coef.copy()
            for j in range(p):
                r = y - X @ coef + X[:, j] * coef[j]
                z = X[:, j] @ r / n
                denom = (X[:, j] @ X[:, j] / n) + l2
                if z > l1:
                    coef[j] = (z - l1) / denom
                elif z < -l1:
                    coef[j] = (z + l1) / denom
                else:
                    coef[j] = 0.0
            if np.max(np.abs(coef - coef_old)) < self.tol:
                break
        self.coef_ = coef
        return self

    def predict(self, X: NDArray) -> NDArray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
