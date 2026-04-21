"""Kernel functions for SVM."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def linear_kernel(X1: NDArray, X2: NDArray) -> NDArray:
    return X1 @ X2.T


def rbf_kernel(X1: NDArray, X2: NDArray, gamma: float = 1.0) -> NDArray:
    """Radial Basis Function (Gaussian) kernel."""
    X1_sq = (X1 ** 2).sum(axis=1, keepdims=True)
    X2_sq = (X2 ** 2).sum(axis=1)
    sq_dist = X1_sq + X2_sq - 2 * X1 @ X2.T
    return np.exp(-gamma * np.maximum(sq_dist, 0))


def poly_kernel(X1: NDArray, X2: NDArray, degree: int = 3,
                coef0: float = 1.0, gamma: float = 1.0) -> NDArray:
    return (gamma * X1 @ X2.T + coef0) ** degree


def sigmoid_kernel(X1: NDArray, X2: NDArray, gamma: float = 1.0, coef0: float = 0.0) -> NDArray:
    return np.tanh(gamma * X1 @ X2.T + coef0)


KERNELS = {
    "linear": linear_kernel,
    "rbf": rbf_kernel,
    "poly": poly_kernel,
    "sigmoid": sigmoid_kernel,
}
