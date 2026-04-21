"""Random Forest for classification and regression."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest using bagging + feature randomness.

    Parameters
    ----------
    n_estimators : int      Number of trees
    task : str              'classification' | 'regression'
    max_depth : int | None
    min_samples_split : int
    min_samples_leaf : int
    max_features : str | int | float   Features per split ('sqrt', 'log2', float, int)
    bootstrap : bool        Sample with replacement
    oob_score : bool        Compute out-of-bag score
    random_state : int | None
    """

    def __init__(self, n_estimators: int = 100, task: str = "classification",
                 max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: str | int | float = "sqrt",
                 bootstrap: bool = True, oob_score: bool = False,
                 random_state: int | None = None):
        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X: NDArray, y: NDArray) -> "RandomForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)
        self.estimators_: list[DecisionTree] = []
        self.n_features_in_ = p

        if self.task == "classification":
            self.classes_ = np.unique(y)

        oob_preds = np.full((n,), np.nan) if self.oob_score else None
        oob_counts = np.zeros(n, dtype=int) if self.oob_score else None
        oob_accum = np.zeros(n) if self.oob_score else None

        for i in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31))
            np.random.seed(seed)
            if self.bootstrap:
                indices = rng.choice(n, size=n, replace=True)
                oob_idx = np.setdiff1d(np.arange(n), indices) if self.oob_score else None
            else:
                indices = np.arange(n)
                oob_idx = None

            tree = DecisionTree(
                task=self.task,
                criterion="gini" if self.task == "classification" else "mse",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)

            if self.oob_score and oob_idx is not None and len(oob_idx) > 0:
                preds = tree.predict(X[oob_idx])
                oob_accum[oob_idx] += preds
                oob_counts[oob_idx] += 1

        if self.oob_score:
            mask = oob_counts > 0
            final = oob_accum[mask] / oob_counts[mask]
            if self.task == "classification":
                self.oob_score_ = float(np.mean(np.round(final) == y[mask]))
            else:
                ss_res = np.sum((y[mask] - final) ** 2)
                ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
                self.oob_score_ = float(1 - ss_res / (ss_tot + 1e-15))

        return self

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        preds = np.array([t.predict(X) for t in self.estimators_])  # (n_est, n)
        if self.task == "classification":
            # Majority vote — safe for any class labels (not just 0-indexed ints)
            n_samples = preds.shape[1]
            result = np.empty(n_samples, dtype=self.classes_.dtype)
            for i in range(n_samples):
                votes = preds[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                result[i] = unique[counts.argmax()]
            return result
        return preds.mean(axis=0)

    def predict_proba(self, X: NDArray) -> NDArray:
        """Classification only: average probability across trees (soft voting)."""
        X = np.asarray(X, dtype=float)
        K = len(self.classes_)
        proba = np.zeros((X.shape[0], K))
        for tree in self.estimators_:
            raw = tree.predict(X).astype(int)
            for k in range(K):
                proba[:, k] += (raw == self.classes_[k]).astype(float)
        return proba / self.n_estimators

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(y_pred == y))
        ss_res = np.sum((np.asarray(y) - y_pred) ** 2)
        ss_tot = np.sum((np.asarray(y) - np.mean(y)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))

    @property
    def feature_importances_(self) -> NDArray:
        imp = np.zeros(self.n_features_in_)
        for tree in self.estimators_:
            imp += tree.feature_importances_
        return imp / self.n_estimators
