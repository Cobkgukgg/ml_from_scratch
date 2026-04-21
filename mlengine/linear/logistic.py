"""Logistic regression with binary and multinomial support."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..utils.validation import check_X_y, check_is_fitted, check_array
from ..utils.math_utils import sigmoid, softmax, add_bias, clip_gradients


class LogisticRegression:
    """
    Binary and multinomial logistic regression with L2 regularisation.

    Parameters
    ----------
    C : float          Inverse of regularisation strength (larger = less reg)
    multi_class : str  'ovr' (one-vs-rest) or 'softmax'
    lr : float         Learning rate
    n_iter : int       Maximum iterations
    tol : float        Convergence threshold on gradient norm
    fit_intercept : bool
    """

    def __init__(self, C: float = 1.0, multi_class: str = "ovr",
                 lr: float = 0.1, n_iter: int = 1000, tol: float = 1e-4,
                 fit_intercept: bool = True):
        self.C = C
        self.multi_class = multi_class
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _fit_binary(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit one binary classifier, return weight vector."""
        n, p = X.shape
        w = np.zeros(p)
        for _ in range(self.n_iter):
            pred = sigmoid(X @ w)
            grad = X.T @ (pred - y) / n + w / self.C
            grad = clip_gradients(grad)
            w -= self.lr * grad
            if np.linalg.norm(grad) < self.tol:
                break
        return w

    def fit(self, X: NDArray, y: NDArray) -> "LogisticRegression":
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = add_bias(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            y_bin = (y == self.classes_[1]).astype(float)
            self.coef_ = self._fit_binary(X, y_bin)[np.newaxis, :]  # (1, p)
        elif self.multi_class == "ovr":
            weights = []
            for c in self.classes_:
                y_bin = (y == c).astype(float)
                weights.append(self._fit_binary(X, y_bin))
            self.coef_ = np.array(weights)  # (K, p)
        else:  # softmax
            n, p = X.shape
            K = n_classes
            W = np.zeros((K, p))
            y_enc = np.zeros((n, K))
            for i, c in enumerate(self.classes_):
                y_enc[y == c, i] = 1.0
            for _ in range(self.n_iter):
                logits = X @ W.T          # (n, K)
                probs = softmax(logits)   # (n, K)
                grad = (probs - y_enc).T @ X / n + W / self.C  # (K, p)
                W -= self.lr * grad
                if np.linalg.norm(grad) < self.tol:
                    break
            self.coef_ = W

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, 0]
            self.coef_ = self.coef_[:, 1:]
        else:
            self.intercept_ = np.zeros(self.coef_.shape[0])
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["coef_"])
        X = check_array(X)
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: NDArray) -> NDArray:
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            p_pos = sigmoid(scores[:, 0])
            return np.column_stack([1 - p_pos, p_pos])
        return softmax(scores)

    def predict(self, X: NDArray) -> NDArray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == y))
