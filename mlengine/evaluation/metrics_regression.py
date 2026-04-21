"""Regression metrics: MSE, RMSE, MAE, R², MAPE, Huber, explained variance."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def mean_squared_error(y_true: NDArray, y_pred: NDArray, squared: bool = True) -> float:
    err = np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
    return float(err if squared else np.sqrt(err))


def root_mean_squared_error(y_true: NDArray, y_pred: NDArray) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)


def mean_absolute_error(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true: NDArray, y_pred: NDArray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-15))


def mean_absolute_percentage_error(y_true: NDArray, y_pred: NDArray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs((y_true - np.asarray(y_pred)) / (np.abs(y_true) + eps))))


def median_absolute_error(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def huber_loss(y_true: NDArray, y_pred: NDArray, delta: float = 1.0) -> float:
    """Huber loss: MSE for small errors, MAE for large ones."""
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    huber = np.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    return float(huber.mean())


def explained_variance_score(y_true: NDArray, y_pred: NDArray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    diff = y_true - np.asarray(y_pred)
    return float(1 - diff.var() / (y_true.var() + 1e-15))


def max_error(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mean_squared_log_error(y_true: NDArray, y_pred: NDArray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert (y_true >= 0).all() and (y_pred >= 0).all(), "Values must be non-negative."
    return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
