"""Categorical encoders: OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class LabelEncoder:
    """Encode string/integer labels as integers 0..n_classes-1."""

    def fit(self, y) -> "LabelEncoder":
        self.classes_ = np.unique(y)
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y) -> NDArray:
        return np.array([self._map[v] for v in y])

    def fit_transform(self, y) -> NDArray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y) -> NDArray:
        return self.classes_[np.asarray(y)]


class OneHotEncoder:
    """
    One-hot encode categorical features.

    Parameters
    ----------
    sparse : bool       Return dense array (sparse not yet implemented)
    drop : str | None   'first' to drop first category (avoid multicollinearity)
    handle_unknown : str 'error' | 'ignore'
    """

    def __init__(self, sparse: bool = False, drop: str | None = None,
                 handle_unknown: str = "error"):
        self.sparse = sparse
        self.drop = drop
        self.handle_unknown = handle_unknown

    def fit(self, X: NDArray, y=None) -> "OneHotEncoder":
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        self.categories_: list[NDArray] = []
        self.drop_idx_: list[int | None] = []
        for j in range(X.shape[1]):
            cats = np.unique(X[:, j])
            self.categories_.append(cats)
            self.drop_idx_.append(0 if self.drop == "first" else None)
        return self

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        parts = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            enc = np.zeros((len(col), len(cats)))
            for i, v in enumerate(col):
                idx = np.where(cats == v)[0]
                if len(idx) == 0:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category: {v}")
                else:
                    enc[i, idx[0]] = 1.0
            if self.drop_idx_[j] is not None:
                enc = np.delete(enc, self.drop_idx_[j], axis=1)
            parts.append(enc)
        return np.hstack(parts)

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)


class OrdinalEncoder:
    """Encode each feature as an integer ordinal."""

    def fit(self, X: NDArray, y=None) -> "OrdinalEncoder":
        X = np.asarray(X)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self._maps = [{v: i for i, v in enumerate(cats)} for cats in self.categories_]
        return self

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X)
        out = np.zeros_like(X, dtype=float)
        for j, m in enumerate(self._maps):
            out[:, j] = [m.get(v, -1) for v in X[:, j]]
        return out

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)


class TargetEncoder:
    """
    Replace each category with the mean of the target within that category.
    Uses smoothing to handle rare categories.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing

    def fit(self, X: NDArray, y: NDArray) -> "TargetEncoder":
        X = np.asarray(X).ravel()
        y = np.asarray(y, dtype=float)
        self.global_mean_ = y.mean()
        self.encoding_: dict = {}
        for cat in np.unique(X):
            mask = X == cat
            n = mask.sum()
            mean = y[mask].mean()
            lam = n / (n + self.smoothing)
            self.encoding_[cat] = lam * mean + (1 - lam) * self.global_mean_
        return self

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X).ravel()
        return np.array([self.encoding_.get(v, self.global_mean_) for v in X])

    def fit_transform(self, X: NDArray, y: NDArray) -> NDArray:
        return self.fit(X, y).transform(X)
