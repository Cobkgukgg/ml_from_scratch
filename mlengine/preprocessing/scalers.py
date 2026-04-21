"""Feature scalers: StandardScaler, MinMaxScaler, RobustScaler, Normalizer."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class StandardScaler:
    """Zero mean, unit variance scaling. Handles zero-variance features."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: NDArray, y=None) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        self.scale_ = X.std(axis=0) if self.with_std else np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: NDArray) -> NDArray:
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_


class MinMaxScaler:
    """Scale features to [feature_range[0], feature_range[1]]."""

    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        self.feature_range = feature_range

    def fit(self, X: NDArray, y=None) -> "MinMaxScaler":
        X = np.asarray(X, dtype=float)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1.0
        return self

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        return (X - self.data_min_) / self.data_range_ * (hi - lo) + lo

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: NDArray) -> NDArray:
        lo, hi = self.feature_range
        return (np.asarray(X, dtype=float) - lo) / (hi - lo) * self.data_range_ + self.data_min_


class RobustScaler:
    """Scale using median and IQR — robust to outliers."""

    def __init__(self, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range: tuple[float, float] = (25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range

    def fit(self, X: NDArray, y=None) -> "RobustScaler":
        X = np.asarray(X, dtype=float)
        self.center_ = np.median(X, axis=0) if self.with_centering else np.zeros(X.shape[1])
        lo, hi = self.quantile_range
        iqr = np.percentile(X, hi, axis=0) - np.percentile(X, lo, axis=0)
        iqr[iqr == 0] = 1.0
        self.scale_ = iqr if self.with_scaling else np.ones(X.shape[1])
        return self

    def transform(self, X: NDArray) -> NDArray:
        return (np.asarray(X, dtype=float) - self.center_) / self.scale_

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)


class Normalizer:
    """Scale each sample (row) to unit norm."""

    def __init__(self, norm: str = "l2"):
        self.norm = norm

    def fit(self, X) -> "Normalizer":
        return self  # stateless

    def transform(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.norm == "l1":
            norms = np.abs(X).sum(axis=1, keepdims=True)
        else:
            norms = np.abs(X).max(axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.transform(X)
