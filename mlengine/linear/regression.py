"""Linear and Ridge/Lasso regression from scratch."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..utils.validation import check_X_y, check_is_fitted, check_array
from ..utils.math_utils import add_bias


class LinearRegression:
    """
    Ordinary Least Squares linear regression.
    Solved analytically via the normal equation or iteratively via gradient descent.

    Parameters
    ----------
    method : {'normal', 'gd'}  normal equation or gradient descent
    lr : float  learning rate (only for method='gd')
    n_iter : int  number of GD iterations
    fit_intercept : bool
    """

    def __init__(self, method: str = "normal", lr: float = 1e-3,
                 n_iter: int = 1000, fit_intercept: bool = True):
        self.method = method
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.losses_: list[float] = []

    def fit(self, X: NDArray, y: NDArray) -> "LinearRegression":
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = add_bias(X)
        n, p = X.shape
        if self.method == "normal":
            # θ = (XᵀX)⁻¹Xᵀy  – use lstsq for numerical stability
            self.coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        else:
            self.coef_ = np.zeros(p)
            for _ in range(self.n_iter):
                residuals = X @ self.coef_ - y
                grad = X.T @ residuals / n
                self.coef_ -= self.lr * grad
                self.losses_.append(float(np.mean(residuals ** 2)))
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["coef_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X: NDArray, y: NDArray) -> float:
        """R² coefficient of determination."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))


class RidgeRegression:
    """L2-regularised linear regression (Ridge). Solved analytically."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: NDArray, y: NDArray) -> "RidgeRegression":
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = add_bias(X)
        n, p = X.shape
        I = np.eye(p)
        if self.fit_intercept:
            I[0, 0] = 0  # don't penalise intercept
        self.coef_ = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["coef_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))


class LassoRegression:
    """
    L1-regularised regression (Lasso) via coordinate descent.
    Converges to the true Lasso solution for convex loss.
    """

    def __init__(self, alpha: float = 1.0, n_iter: int = 1000,
                 tol: float = 1e-4, fit_intercept: bool = True):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    @staticmethod
    def _soft_threshold(z: float, gamma: float) -> float:
        if z > gamma:
            return z - gamma
        if z < -gamma:
            return z + gamma
        return 0.0

    def fit(self, X: NDArray, y: NDArray) -> "LassoRegression":
        X, y = check_X_y(X, y)
        n, p = X.shape
        if self.fit_intercept:
            self.intercept_ = y.mean()
            y = y - self.intercept_
        else:
            self.intercept_ = 0.0

        coef = np.zeros(p)
        for _ in range(self.n_iter):
            coef_old = coef.copy()
            for j in range(p):
                r = y - X @ coef + X[:, j] * coef[j]
                z = X[:, j] @ r / n
                coef[j] = self._soft_threshold(z, self.alpha) / (X[:, j] @ X[:, j] / n)
            if np.max(np.abs(coef - coef_old)) < self.tol:
                break
        self.coef_ = coef
        return self

    def predict(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["coef_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))
