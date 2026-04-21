"""Gradient-based optimizers: SGD, Momentum, RMSProp, Adam, AdaGrad."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, lr: float = 1e-3):
        self.lr = lr
        self._step = 0

    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> dict[str, NDArray]:
        raise NotImplementedError

    def step(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> dict[str, NDArray]:
        self._step += 1
        return self.update(params, grads)


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional Nesterov momentum."""

    def __init__(self, lr: float = 1e-2, momentum: float = 0.0, nesterov: bool = False,
                 weight_decay: float = 0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._velocity: dict[str, NDArray] = {}

    def update(self, params, grads):
        updated = {}
        for k, p in params.items():
            g = grads[k].copy()
            if self.weight_decay:
                g += self.weight_decay * p
            v = self._velocity.get(k, np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            self._velocity[k] = v
            if self.nesterov:
                updated[k] = p + self.momentum * v - self.lr * g
            else:
                updated[k] = p + v
        return updated


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2015)."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m: dict[str, NDArray] = {}
        self._v: dict[str, NDArray] = {}

    def update(self, params, grads):
        updated = {}
        t = self._step
        for k, p in params.items():
            g = grads[k].copy()
            if self.weight_decay:
                g += self.weight_decay * p
            m = self.beta1 * self._m.get(k, np.zeros_like(p)) + (1 - self.beta1) * g
            v = self.beta2 * self._v.get(k, np.zeros_like(p)) + (1 - self.beta2) * g ** 2
            self._m[k] = m
            self._v[k] = v
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            updated[k] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated


class RMSProp(Optimizer):
    """RMSProp optimizer."""

    def __init__(self, lr: float = 1e-3, decay: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.decay = decay
        self.eps = eps
        self._cache: dict[str, NDArray] = {}

    def update(self, params, grads):
        updated = {}
        for k, p in params.items():
            g = grads[k]
            cache = self.decay * self._cache.get(k, np.zeros_like(p)) + (1 - self.decay) * g ** 2
            self._cache[k] = cache
            updated[k] = p - self.lr * g / (np.sqrt(cache) + self.eps)
        return updated


class AdaGrad(Optimizer):
    """AdaGrad optimizer."""

    def __init__(self, lr: float = 1e-2, eps: float = 1e-8):
        super().__init__(lr)
        self.eps = eps
        self._G: dict[str, NDArray] = {}

    def update(self, params, grads):
        updated = {}
        for k, p in params.items():
            g = grads[k]
            G = self._G.get(k, np.zeros_like(p)) + g ** 2
            self._G[k] = G
            updated[k] = p - self.lr * g / (np.sqrt(G) + self.eps)
        return updated
