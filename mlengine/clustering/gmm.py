"""Gaussian Mixture Model via Expectation-Maximisation."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal


class GaussianMixture:
    """
    Gaussian Mixture Model using the EM algorithm.

    Parameters
    ----------
    n_components : int      Number of mixture components
    covariance_type : str   'full' | 'diag' | 'spherical'
    max_iter : int
    tol : float             Log-likelihood convergence tolerance
    reg_covar : float       Regularisation added to covariance diagonals
    n_init : int            Restarts; best log-likelihood is kept
    random_state : int | None
    """

    def __init__(self, n_components: int = 1, covariance_type: str = "full",
                 max_iter: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6,
                 n_init: int = 1, random_state: int | None = None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

    def _init_params(self, X: NDArray, rng) -> None:
        n, d = X.shape
        K = self.n_components
        idx = rng.choice(n, K, replace=False)
        self.means_ = X[idx].copy()
        self.weights_ = np.ones(K) / K
        if self.covariance_type == "full":
            self.covariances_ = np.array([np.eye(d) for _ in range(K)])
        elif self.covariance_type == "diag":
            self.covariances_ = np.ones((K, d))
        else:  # spherical
            self.covariances_ = np.ones(K)

    def _e_step(self, X: NDArray) -> NDArray:
        """Returns responsibilities (n, K)."""
        n = X.shape[0]
        K = self.n_components
        log_resp = np.zeros((n, K))
        for k in range(K):
            if self.covariance_type == "full":
                cov = self.covariances_[k]
            elif self.covariance_type == "diag":
                cov = np.diag(self.covariances_[k])
            else:
                cov = np.eye(X.shape[1]) * self.covariances_[k]
            log_resp[:, k] = np.log(self.weights_[k] + 1e-300) + \
                multivariate_normal.logpdf(X, mean=self.means_[k], cov=cov)
        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True) + 1e-300
        return resp

    def _m_step(self, X: NDArray, resp: NDArray) -> None:
        n, d = X.shape
        K = self.n_components
        Nk = resp.sum(axis=0) + 1e-10
        self.weights_ = Nk / n
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]
        for k in range(K):
            diff = X - self.means_[k]
            if self.covariance_type == "full":
                cov = (resp[:, k, np.newaxis] * diff).T @ diff / Nk[k]
                cov += np.eye(d) * self.reg_covar
                self.covariances_[k] = cov
            elif self.covariance_type == "diag":
                var = (resp[:, k] * (diff ** 2).sum(axis=1)).sum() / (Nk[k] * d)
                self.covariances_[k] = np.maximum(var, self.reg_covar)
            else:
                var = (resp[:, k] * (diff ** 2).sum(axis=1)).sum() / (Nk[k] * d)
                self.covariances_[k] = max(var, self.reg_covar)

    def _log_likelihood(self, X: NDArray) -> float:
        K = self.n_components
        ll = np.zeros(X.shape[0])
        for k in range(K):
            if self.covariance_type == "full":
                cov = self.covariances_[k]
            elif self.covariance_type == "diag":
                cov = np.diag(self.covariances_[k])
            else:
                cov = np.eye(X.shape[1]) * self.covariances_[k]
            ll += self.weights_[k] * np.exp(
                multivariate_normal.logpdf(X, mean=self.means_[k], cov=cov)
            )
        return float(np.sum(np.log(ll + 1e-300)))

    def _fit_once(self, X: NDArray, rng):
        self._init_params(X, rng)
        prev_ll = -np.inf
        for i in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self._log_likelihood(X)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return ll

    def fit(self, X: NDArray) -> "GaussianMixture":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best_ll = -np.inf
        best_params = None
        for _ in range(self.n_init):
            ll = self._fit_once(X, rng)
            if ll > best_ll:
                best_ll = ll
                best_params = (self.means_.copy(), self.covariances_.copy(),
                               self.weights_.copy())
        self.means_, self.covariances_, self.weights_ = best_params
        self.lower_bound_ = best_ll
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        return self._e_step(np.asarray(X, dtype=float))

    def predict(self, X: NDArray) -> NDArray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: NDArray) -> float:
        return self._log_likelihood(np.asarray(X, dtype=float)) / X.shape[0]

    def sample(self, n_samples: int = 1) -> tuple[NDArray, NDArray]:
        """Generate random samples. Returns (X, component_labels)."""
        rng = np.random.default_rng()
        K = self.n_components
        n_per = rng.multinomial(n_samples, self.weights_)
        X_list, y_list = [], []
        for k, nk in enumerate(n_per):
            if nk == 0:
                continue
            if self.covariance_type == "full":
                cov = self.covariances_[k]
            elif self.covariance_type == "diag":
                cov = np.diag(self.covariances_[k])
            else:
                cov = np.eye(self.means_.shape[1]) * self.covariances_[k]
            X_list.append(rng.multivariate_normal(self.means_[k], cov, size=nk))
            y_list.append(np.full(nk, k))
        return np.vstack(X_list), np.concatenate(y_list)
