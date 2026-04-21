"""Input validation helpers."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def check_array(X, ensure_2d: bool = True, dtype=np.float64) -> NDArray:
    X = np.array(X, dtype=dtype)
    if ensure_2d and X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2 and ensure_2d:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array.")
    return X


def check_X_y(X, y):
    X = check_array(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent number of samples: {X.shape[0]} vs {y.shape[0]}"
        )
    return X, y


def check_is_fitted(estimator, attributes: list[str] | None = None):
    if attributes is None:
        attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]
        if not attrs:
            raise RuntimeError(
                f"This {type(estimator).__name__} instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
    else:
        missing = [a for a in attributes if not hasattr(estimator, a)]
        if missing:
            raise RuntimeError(
                f"This {type(estimator).__name__} instance is not fitted. "
                f"Missing attributes: {missing}"
            )


def check_classification_targets(y: NDArray) -> NDArray:
    y = np.asarray(y)
    classes = np.unique(y)
    return classes


def validate_sample_weights(sample_weight, n_samples: int) -> NDArray:
    if sample_weight is None:
        return np.ones(n_samples)
    w = np.asarray(sample_weight, dtype=float)
    if w.shape[0] != n_samples:
        raise ValueError("sample_weight must have same length as X.")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")
    total = w.sum()
    if total == 0:
        raise ValueError("sample_weight sum must be positive.")
    return w / total * n_samples
