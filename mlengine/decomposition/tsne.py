"""
t-SNE (t-Distributed Stochastic Neighbour Embedding).

Reference: van der Maaten & Hinton (2008).
Implementation uses the exact (non-Barnes-Hut) O(n²) algorithm with:
  - Perplexity-based bandwidth search (binary search on entropy)
  - Early exaggeration
  - Momentum scheduling
  - Gradient clipping
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class TSNE:
    """
    t-SNE dimensionality reduction.

    Parameters
    ----------
    n_components : int      Output dimensions (usually 2 or 3)
    perplexity : float      Effective number of nearest neighbours (~5–50)
    learning_rate : float   Step size ('auto' sets n/12 clamped to [10,1000])
    n_iter : int            Total gradient-descent iterations
    early_exaggeration : float  Multiplier on P in early phase
    n_iter_early_exag : int     How many iters to keep early exaggeration
    momentum : float        Base momentum
    final_momentum : float  Momentum after switch_iter
    switch_iter : int       When to switch from base to final momentum
    random_state : int | None
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: float | str = "auto", n_iter: int = 1000,
                 early_exaggeration: float = 12.0, n_iter_early_exag: int = 250,
                 momentum: float = 0.5, final_momentum: float = 0.8,
                 switch_iter: int = 250, random_state: int | None = None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.n_iter_early_exag = n_iter_early_exag
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.switch_iter = switch_iter
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _pairwise_sq_dist(self, X: NDArray) -> NDArray:
        sq = (X ** 2).sum(axis=1)
        return np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)

    def _joint_probabilities(self, D2: NDArray) -> NDArray:
        """Compute symmetric joint probabilities P from squared distances."""
        n = D2.shape[0]
        P = np.zeros((n, n))
        log_perp = np.log2(self.perplexity)

        for i in range(n):
            di = D2[i].copy()
            di[i] = np.inf
            beta_min, beta_max = -np.inf, np.inf
            beta = 1.0
            for _ in range(50):
                exp_d = np.exp(-beta * di)
                exp_d[i] = 0.0
                S = exp_d.sum() + 1e-10
                # Zero out di[i] to avoid inf*0=nan; exp_d[i] is already 0
                di_safe = di.copy()
                di_safe[i] = 0.0
                H = np.log2(S) + beta * (di_safe * exp_d).sum() / S
                Hdiff = H - log_perp
                if abs(Hdiff) < 1e-5:
                    break
                if Hdiff > 0:
                    beta_min = beta
                    beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
                else:
                    beta_max = beta
                    beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
            P[i] = exp_d / S


        P = (P + P.T) / (2 * n)
        np.fill_diagonal(P, 0)
        P = np.maximum(P, 1e-12)
        return P

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        lr = n / 12.0 if self.learning_rate == "auto" else float(self.learning_rate)
        lr = np.clip(lr, 10.0, 1000.0)

        D2 = self._pairwise_sq_dist(X)
        P = self._joint_probabilities(D2)
        P *= self.early_exaggeration
        # Ensure early exaggeration ends before the final iteration
        n_iter_early_exag = min(self.n_iter_early_exag, self.n_iter - 1)

        Y = rng.normal(0, 1e-4, (n, self.n_components))
        velocity = np.zeros_like(Y)
        self.kl_divergences_: list[float] = []

        for t in range(1, self.n_iter + 1):
            if t == n_iter_early_exag + 1:
                P /= self.early_exaggeration
            mom = self.momentum if t < self.switch_iter else self.final_momentum

            # Q distribution (Student-t with df=1)
            Dy2 = self._pairwise_sq_dist(Y)
            inv = 1.0 / (1.0 + Dy2)
            np.fill_diagonal(inv, 0)
            Q = inv / (inv.sum() + 1e-10)
            Q = np.maximum(Q, 1e-12)

            # Gradient
            PQ = P - Q
            grad = np.zeros_like(Y)
            for i in range(n):
                diff = Y[i] - Y           # (n, d)
                grad[i] = 4 * (PQ[i, :, None] * diff * inv[i, :, None]).sum(axis=0)

            grad = np.clip(grad, -10.0, 10.0)
            velocity = mom * velocity - lr * grad
            Y += velocity

            if t % 50 == 0 or t == self.n_iter:
                kl = float(np.sum(P[P > 0] * np.log(P[P > 0] / Q[P > 0])))
                self.kl_divergences_.append(kl)

        self.embedding_ = Y
        return Y

    def fit(self, X: NDArray, y=None) -> "TSNE":
        self.fit_transform(X)
        return self
