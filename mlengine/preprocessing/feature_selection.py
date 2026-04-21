"""Feature selection: variance threshold, univariate, RFE, mutual information."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class VarianceThreshold:
    """Remove features with variance below a threshold."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X: NDArray) -> "VarianceThreshold":
        X = np.asarray(X, dtype=float)
        self.variances_ = X.var(axis=0)
        self.support_ = self.variances_ > self.threshold
        return self

    def transform(self, X: NDArray) -> NDArray:
        return np.asarray(X, dtype=float)[:, self.support_]

    def fit_transform(self, X: NDArray, y=None) -> NDArray:
        return self.fit(X).transform(X)

    def get_support(self) -> NDArray:
        return self.support_


class SelectKBest:
    """
    Select features by highest score.

    Parameters
    ----------
    score_func : callable  f(X, y) -> (scores, pvalues) or just scores
    k : int | 'all'
    """

    def __init__(self, score_func=None, k: int | str = 10):
        self.score_func = score_func if score_func is not None else f_classif
        self.k = k

    def fit(self, X: NDArray, y: NDArray) -> "SelectKBest":
        X = np.asarray(X, dtype=float)
        result = self.score_func(X, y)
        self.scores_ = result[0] if isinstance(result, tuple) else result
        self.pvalues_ = result[1] if isinstance(result, tuple) else None
        k = X.shape[1] if self.k == "all" else min(self.k, X.shape[1])
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        top_k = np.argsort(self.scores_)[-k:]
        self.support_[top_k] = True
        return self

    def transform(self, X: NDArray) -> NDArray:
        return np.asarray(X, dtype=float)[:, self.support_]

    def fit_transform(self, X: NDArray, y: NDArray) -> NDArray:
        return self.fit(X, y).transform(X)


def f_classif(X: NDArray, y: NDArray):
    """ANOVA F-statistic for classification."""
    X = np.asarray(X, dtype=float)
    classes = np.unique(y)
    n = X.shape[0]
    grand_mean = X.mean(axis=0)
    ss_between = sum(
        np.sum(y == c) * (X[y == c].mean(axis=0) - grand_mean) ** 2
        for c in classes
    )
    ss_within = sum(
        np.sum((X[y == c] - X[y == c].mean(axis=0)) ** 2, axis=0)
        for c in classes
    )
    df_between = len(classes) - 1
    df_within = n - len(classes)
    F = (ss_between / df_between) / (ss_within / df_within + 1e-10)
    return F, None


def mutual_info_classif(X: NDArray, y: NDArray, n_bins: int = 10):
    """Estimate mutual information between each feature and target class."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    mi = np.zeros(p)
    y = np.asarray(y)
    for j in range(p):
        col = X[:, j]
        bins = np.histogram_bin_edges(col, bins=n_bins)
        x_disc = np.digitize(col, bins[:-1]) - 1
        classes = np.unique(y)
        x_vals = np.unique(x_disc)
        px = np.array([(x_disc == v).mean() for v in x_vals])
        py = np.array([(y == c).mean() for c in classes])
        for xi, xv in enumerate(x_vals):
            for yi, yc in enumerate(classes):
                pxy = np.mean((x_disc == xv) & (y == yc))
                if pxy > 0:
                    mi[j] += pxy * np.log(pxy / (px[xi] * py[yi] + 1e-12))
    return np.maximum(mi, 0), None


class RFE:
    """
    Recursive Feature Elimination.
    Iteratively removes the feature with lowest importance.

    Parameters
    ----------
    estimator : object with fit() and feature_importances_ or coef_
    n_features_to_select : int | float
    step : int | float   Number/fraction of features to remove per iteration
    """

    def __init__(self, estimator, n_features_to_select: int | float = 0.5, step: int | float = 1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step

    def fit(self, X: NDArray, y: NDArray) -> "RFE":
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        target = max(1, int(n_features * self.n_features_to_select)
                     if isinstance(self.n_features_to_select, float)
                     else self.n_features_to_select)
        step = max(1, int(n_features * self.step) if isinstance(self.step, float) else self.step)

        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        current_features = np.where(support)[0]

        while len(current_features) > target:
            self.estimator.fit(X[:, support], y)
            if hasattr(self.estimator, "feature_importances_"):
                importance = self.estimator.feature_importances_
            elif hasattr(self.estimator, "coef_"):
                importance = np.abs(self.estimator.coef_).ravel()
            else:
                break
            n_remove = min(step, len(current_features) - target)
            worst = np.argsort(importance)[:n_remove]
            actual_worst = current_features[worst]
            support[actual_worst] = False
            ranking[actual_worst] = len(current_features) - n_remove + 1
            current_features = np.where(support)[0]
            if len(current_features) <= target:
                break

        self.support_ = support
        self.ranking_ = ranking
        self.estimator.fit(X[:, support], y)
        return self

    def transform(self, X: NDArray) -> NDArray:
        return np.asarray(X, dtype=float)[:, self.support_]

    def fit_transform(self, X: NDArray, y: NDArray) -> NDArray:
        return self.fit(X, y).transform(X)
