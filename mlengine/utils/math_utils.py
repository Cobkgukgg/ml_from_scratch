"""Core mathematical utilities used throughout mlengine."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray) -> NDArray:
    """Numerically stable sigmoid function."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def softmax(z: NDArray, axis: int = -1) -> NDArray:
    """Numerically stable softmax."""
    z = z - z.max(axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=axis, keepdims=True)


def log_softmax(z: NDArray, axis: int = -1) -> NDArray:
    """Numerically stable log-softmax."""
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def relu(z: NDArray) -> NDArray:
    return np.maximum(0, z)


def tanh(z: NDArray) -> NDArray:
    return np.tanh(z)


def euclidean_distance(a: NDArray, b: NDArray) -> NDArray:
    """Pairwise Euclidean distances. a: (m, d), b: (n, d) -> (m, n)."""
    a2 = (a ** 2).sum(axis=1, keepdims=True)
    b2 = (b ** 2).sum(axis=1)
    return np.sqrt(np.maximum(a2 + b2 - 2 * a @ b.T, 0))


def cosine_similarity(a: NDArray, b: NDArray) -> NDArray:
    """Pairwise cosine similarity. a: (m, d), b: (n, d) -> (m, n)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def entropy(p: NDArray, eps: float = 1e-12) -> float:
    """Shannon entropy of probability distribution p."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def gini_impurity(y: NDArray) -> float:
    """Gini impurity for a 1-D label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs ** 2))


def information_gain(y: NDArray, y_left: NDArray, y_right: NDArray) -> float:
    """Information gain using entropy."""
    n = len(y)
    if n == 0:
        return 0.0
    return entropy(np.bincount(y) / n) - (
        len(y_left) / n * entropy(np.bincount(y_left) / len(y_left)) if len(y_left) else 0
    ) - (
        len(y_right) / n * entropy(np.bincount(y_right) / len(y_right)) if len(y_right) else 0
    )


def add_bias(X: NDArray) -> NDArray:
    """Prepend a column of ones to X."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def clip_gradients(grad: NDArray, max_norm: float = 5.0) -> NDArray:
    """Global gradient clipping by norm."""
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * max_norm / (norm + 1e-8)
    return grad


def numerical_gradient(f, x: NDArray, eps: float = 1e-5) -> NDArray:
    """Central finite differences for gradient checking."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    x = x.astype(float)
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = old
        it.iternext()
    return grad
