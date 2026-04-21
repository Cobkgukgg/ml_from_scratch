"""Learning rate schedulers."""

from __future__ import annotations
import numpy as np


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._step = 0

    def step(self):
        self._step += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError


class StepLR(LRScheduler):
    """Decay LR by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size: int = 10, gamma: float = 0.1):
        super().__init__(optimizer)
        self._base_lr = optimizer.lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self._base_lr * (self.gamma ** (self._step // self.step_size))


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma: float = 0.95):
        super().__init__(optimizer)
        self._base_lr = optimizer.lr
        self.gamma = gamma

    def get_lr(self) -> float:
        return self._base_lr * (self.gamma ** self._step)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing without warm restarts."""

    def __init__(self, optimizer, T_max: int = 100, eta_min: float = 0.0):
        super().__init__(optimizer)
        self._base_lr = optimizer.lr
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        t = self._step % (2 * self.T_max)
        return self.eta_min + 0.5 * (self._base_lr - self.eta_min) * (
            1 + np.cos(np.pi * t / self.T_max)
        )


class WarmupScheduler(LRScheduler):
    """Linear warmup then hand-off to another scheduler."""

    def __init__(self, optimizer, warmup_steps: int, after: LRScheduler):
        super().__init__(optimizer)
        self._base_lr = optimizer.lr
        self.warmup_steps = warmup_steps
        self.after = after

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            self.optimizer.lr = self._base_lr * self._step / self.warmup_steps
        else:
            self.after.step()

    def get_lr(self) -> float:
        return self.optimizer.lr
