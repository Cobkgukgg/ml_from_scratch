"""Batch, mini-batch, and stochastic gradient descent training loops."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Callable


def mini_batch_generator(X: NDArray, y: NDArray, batch_size: int, shuffle: bool = True):
    """Yield (X_batch, y_batch) mini-batches."""
    n = X.shape[0]
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = indices[start: start + batch_size]
        yield X[idx], y[idx]


def run_epoch(
    X: NDArray,
    y: NDArray,
    loss_fn: Callable,
    grad_fn: Callable,
    params: dict,
    optimizer,
    batch_size: int = 32,
) -> float:
    """Run one epoch and return mean loss."""
    losses = []
    for Xb, yb in mini_batch_generator(X, y, batch_size):
        loss = loss_fn(params, Xb, yb)
        grads = grad_fn(params, Xb, yb)
        params.update(optimizer.step(params, grads))
        losses.append(loss)
    return float(np.mean(losses))
