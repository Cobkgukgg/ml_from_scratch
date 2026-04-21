"""Gradient Boosting for classification and regression (GBDT)."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .decision_tree import DecisionTree
from ..utils.math_utils import sigmoid


class GradientBoostingRegressor:
    """
    Gradient Boosted Decision Trees for regression (squared loss).

    Builds additive model: F_m(x) = F_{m-1}(x) + lr * h_m(x)
    where h_m fits pseudo-residuals -∂L/∂F_{m-1}.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 subsample: float = 1.0, random_state: int | None = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: NDArray, y: NDArray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        self.F0_ = y.mean()
        F = np.full(n, self.F0_)
        self.estimators_: list[DecisionTree] = []
        self.train_losses_: list[float] = []

        for _ in range(self.n_estimators):
            residuals = y - F  # negative gradient of MSE
            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)
            tree = DecisionTree(task="regression", max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            tree.fit(X[idx], residuals[idx])
            update = tree.predict(X)
            F += self.learning_rate * update
            self.estimators_.append(tree)
            self.train_losses_.append(float(np.mean((y - F) ** 2)))

        return self

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.F0_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X, y) -> float:
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))


class GradientBoostingClassifier:
    """
    GBDT for binary classification (log-loss / deviance).
    Uses log-odds initial prediction and fits log-odds residuals.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 subsample: float = 1.0, random_state: int | None = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: NDArray, y: NDArray) -> "GradientBoostingClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        p0 = np.clip(y.mean(), 1e-7, 1 - 1e-7)
        self.F0_ = float(np.log(p0 / (1 - p0)))  # log-odds
        F = np.full(n, self.F0_)
        self.estimators_: list[DecisionTree] = []
        self.train_losses_: list[float] = []

        for _ in range(self.n_estimators):
            probs = sigmoid(F)
            residuals = y - probs   # negative gradient of log-loss
            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)
            tree = DecisionTree(task="regression", max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            tree.fit(X[idx], residuals[idx])
            update = tree.predict(X)
            F += self.learning_rate * update
            self.estimators_.append(tree)
            probs = sigmoid(F)
            loss = -np.mean(y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15))
            self.train_losses_.append(float(loss))

        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.F0_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        p = sigmoid(F)
        return np.column_stack([1 - p, p])

    def predict(self, X: NDArray) -> NDArray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
