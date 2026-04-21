"""
Support Vector Machine via Sequential Minimal Optimization (SMO).

Reference: John Platt (1998) "Sequential Minimal Optimization:
A Fast Algorithm for Training Support Vector Machines"
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .kernel import KERNELS


class SVC:
    """
    Binary SVM classifier using SMO.

    Parameters
    ----------
    C : float       Regularization (margin softness)
    kernel : str    'linear' | 'rbf' | 'poly' | 'sigmoid'
    gamma : float   Kernel coefficient for rbf/poly/sigmoid
    degree : int    Polynomial degree
    coef0 : float   Independent term in poly/sigmoid
    tol : float     KKT tolerance
    max_iter : int  Maximum SMO passes
    """

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: float = 1.0,
                 degree: int = 3, coef0: float = 1.0, tol: float = 1e-3,
                 max_iter: int = 200):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    def _kernel(self, X1, X2):
        fn = KERNELS[self.kernel]
        kwargs = {}
        if self.kernel in ("rbf", "poly", "sigmoid"):
            kwargs["gamma"] = self.gamma
        if self.kernel in ("poly", "sigmoid"):
            kwargs["coef0"] = self.coef0
        if self.kernel == "poly":
            kwargs["degree"] = self.degree
        return fn(X1, X2, **kwargs)

    def _decision(self, i):
        return float(np.sum(self.alpha_ * self.y_train_ * self.K_[:, i]) + self.b_)

    def fit(self, X: NDArray, y: NDArray) -> "SVC":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if not set(np.unique(y)).issubset({-1.0, 1.0, -1, 1}):
            raise ValueError("SVC requires binary labels: -1 and +1. Got: " + str(np.unique(y)))
        n = X.shape[0]
        self.X_train_ = X
        self.y_train_ = y
        self.alpha_ = np.zeros(n)
        self.b_ = 0.0
        self.K_ = self._kernel(X, X)

        passes = 0
        while passes < self.max_iter:
            changed = 0
            for i in range(n):
                Ei = self._decision(i) - y[i]
                if (y[i] * Ei < -self.tol and self.alpha_[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alpha_[i] > 0):
                    # Select j randomly != i
                    j = i
                    while j == i:
                        j = np.random.randint(n)
                    Ej = self._decision(j) - y[j]

                    ai_old, aj_old = self.alpha_[i], self.alpha_[j]
                    if y[i] != y[j]:
                        L = max(0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)
                    if L >= H:
                        continue

                    eta = 2 * self.K_[i, j] - self.K_[i, i] - self.K_[j, j]
                    if eta >= 0:
                        continue

                    self.alpha_[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha_[j] = np.clip(self.alpha_[j], L, H)
                    if abs(self.alpha_[j] - aj_old) < 1e-5:
                        continue

                    self.alpha_[i] += y[i] * y[j] * (aj_old - self.alpha_[j])
                    b1 = self.b_ - Ei - y[i] * (self.alpha_[i] - ai_old) * self.K_[i, i] \
                         - y[j] * (self.alpha_[j] - aj_old) * self.K_[i, j]
                    b2 = self.b_ - Ej - y[i] * (self.alpha_[i] - ai_old) * self.K_[i, j] \
                         - y[j] * (self.alpha_[j] - aj_old) * self.K_[j, j]
                    if 0 < self.alpha_[i] < self.C:
                        self.b_ = b1
                    elif 0 < self.alpha_[j] < self.C:
                        self.b_ = b2
                    else:
                        self.b_ = (b1 + b2) / 2
                    changed += 1

            passes = passes + 1 if changed == 0 else 0

        self.support_vectors_ = X[self.alpha_ > 1e-5]
        self.support_alpha_ = self.alpha_[self.alpha_ > 1e-5]
        self.support_y_ = y[self.alpha_ > 1e-5]
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        K = self._kernel(self.X_train_, X)  # (n_train, n_test)
        return (self.alpha_ * self.y_train_) @ K + self.b_

    def predict(self, X: NDArray) -> NDArray:
        return np.sign(self.decision_function(X))

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class SVR:
    """Support Vector Regression using epsilon-insensitive loss + SMO."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, kernel: str = "rbf",
                 gamma: float = 1.0, tol: float = 1e-3, max_iter: int = 200):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

    def _kernel(self, X1, X2):
        return KERNELS[self.kernel](X1, X2, gamma=self.gamma)

    def fit(self, X: NDArray, y: NDArray) -> "SVR":
        """Reduce to SVC on doubled dataset (standard SVR primal-dual reformulation)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        self.X_train_ = X
        self.y_train_ = y
        # Use simple gradient projection for SVR
        alpha_pos = np.zeros(n)
        alpha_neg = np.zeros(n)
        K = self._kernel(X, X)
        self.b_ = 0.0

        for _ in range(self.max_iter):
            F = (alpha_pos - alpha_neg) @ K + self.b_
            for i in range(n):
                err = F[i] - y[i]
                # Update alpha_pos
                da = np.clip(-(err + self.epsilon) / (K[i, i] + 1e-8), -alpha_pos[i], self.C - alpha_pos[i])
                alpha_pos[i] += da
                F += da * K[i]
                # Update alpha_neg
                da = np.clip((err - self.epsilon) / (K[i, i] + 1e-8), -alpha_neg[i], self.C - alpha_neg[i])
                alpha_neg[i] += da
                F -= da * K[i]
            sv_mask = (alpha_pos + alpha_neg) > 1e-5
            if sv_mask.any():
                self.b_ = np.mean(y[sv_mask] - F[sv_mask])

        self.alpha_ = alpha_pos - alpha_neg
        return self

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        K = self._kernel(self.X_train_, X)
        return self.alpha_ @ K + self.b_

    def score(self, X, y) -> float:
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))
