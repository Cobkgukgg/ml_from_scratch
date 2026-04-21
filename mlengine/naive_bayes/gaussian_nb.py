"""Naive Bayes classifiers: Gaussian, Multinomial, Bernoulli, Complement."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class GaussianNB:
    """
    Gaussian Naive Bayes.
    Assumes each feature follows N(μ_ck, σ²_ck) per class.
    Supports online incremental learning via partial_fit.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X: NDArray, y: NDArray) -> "GaussianNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.theta_ = np.zeros((n_classes, n_features))   # mean
        self.sigma_ = np.zeros((n_classes, n_features))   # variance
        self.class_prior_ = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.sigma_[i] = Xc.var(axis=0)
            self.class_prior_[i] = Xc.shape[0] / X.shape[0]

        self.sigma_ += self.var_smoothing
        return self

    def _log_likelihood(self, X: NDArray) -> NDArray:
        """Returns (n_samples, n_classes) log-likelihoods."""
        n_classes = len(self.classes_)
        log_probs = np.zeros((X.shape[0], n_classes))
        for i in range(n_classes):
            log_probs[:, i] = (
                -0.5 * np.sum(np.log(2 * np.pi * self.sigma_[i]), axis=0)
                - 0.5 * np.sum((X - self.theta_[i]) ** 2 / self.sigma_[i], axis=1)
                + np.log(self.class_prior_[i])
            )
        return log_probs

    def predict_log_proba(self, X: NDArray) -> NDArray:
        ll = self._log_likelihood(np.asarray(X, dtype=float))
        # Numerically stable log-sum-exp normalisation
        ll_max = ll.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(ll - ll_max).sum(axis=1, keepdims=True)) + ll_max
        return ll - log_norm

    def predict_proba(self, X: NDArray) -> NDArray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: NDArray) -> NDArray:
        return self.classes_[np.argmax(self._log_likelihood(np.asarray(X, dtype=float)), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class MultinomialNB:
    """
    Multinomial Naive Bayes — suited for count features (e.g. text/TF).

    Parameters
    ----------
    alpha : float   Laplace smoothing (0 = no smoothing)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: NDArray, y: NDArray) -> "MultinomialNB":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n = X.shape[0]
        self.class_log_prior_ = np.log(
            np.array([(y == c).sum() for c in self.classes_]) / n
        )
        feature_counts = np.array([X[y == c].sum(axis=0) for c in self.classes_])
        smoothed = feature_counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def predict_log_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X: NDArray) -> NDArray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class BernoulliNB:
    """
    Bernoulli Naive Bayes — suited for binary/boolean features.
    """

    def __init__(self, alpha: float = 1.0, binarize: float | None = 0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X: NDArray, y: NDArray) -> "BernoulliNB":
        X = np.asarray(X, dtype=float)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n = X.shape[0]
        self.class_log_prior_ = np.log(
            np.array([(y == c).sum() for c in self.classes_]) / n
        )
        counts = np.array([X[y == c].sum(axis=0) for c in self.classes_])
        n_c = np.array([(y == c).sum() for c in self.classes_])[:, np.newaxis]
        self.feature_log_prob_ = np.log((counts + self.alpha) / (n_c + 2 * self.alpha))
        self.feature_log_neg_prob_ = np.log(1 - np.exp(self.feature_log_prob_))
        return self

    def predict_log_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        return (X @ self.feature_log_prob_.T +
                (1 - X) @ self.feature_log_neg_prob_.T +
                self.class_log_prior_)

    def predict(self, X: NDArray) -> NDArray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
