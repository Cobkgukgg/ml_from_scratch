"""Cross-validation: KFold, StratifiedKFold, cross_val_score, GridSearchCV."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Iterator
import copy


class KFold:
    """
    K-Fold cross-validator.

    Parameters
    ----------
    n_splits : int      Number of folds
    shuffle : bool      Shuffle samples before splitting
    random_state : int | None
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int | None = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: NDArray, y: NDArray | None = None) -> Iterator[tuple[NDArray, NDArray]]:
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, val_idx
            current = stop

    def get_n_splits(self) -> int:
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold: preserves class proportions in each fold.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int | None = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: NDArray, y: NDArray) -> Iterator[tuple[NDArray, NDArray]]:
        y = np.asarray(y)
        classes, y_indices = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state)

        # Collect per-class indices
        per_class = [np.where(y == c)[0] for c in classes]
        if self.shuffle:
            for idx in per_class:
                rng.shuffle(idx)

        # Assign each sample to a fold
        fold_assignment = np.empty(len(y), dtype=int)
        for idx in per_class:
            n = len(idx)
            fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
            fold_sizes[:n % self.n_splits] += 1
            current = 0
            for fold_id, fold_size in enumerate(fold_sizes):
                fold_assignment[idx[current:current + fold_size]] = fold_id
                current += fold_size

        for fold_id in range(self.n_splits):
            val_idx = np.where(fold_assignment == fold_id)[0]
            train_idx = np.where(fold_assignment != fold_id)[0]
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class LeaveOneOut:
    """Leave-One-Out cross-validator (n_splits = n_samples)."""

    def split(self, X: NDArray, y: NDArray | None = None) -> Iterator[tuple[NDArray, NDArray]]:
        n = len(X)
        for i in range(n):
            yield np.concatenate([np.arange(i), np.arange(i + 1, n)]), np.array([i])

    def get_n_splits(self, X: NDArray) -> int:
        return len(X)


def cross_val_score(estimator, X: NDArray, y: NDArray,
                    cv=5, scoring: str = "accuracy") -> NDArray:
    """
    Evaluate estimator with cross-validation.

    Parameters
    ----------
    estimator : object with fit() and score() or predict()
    X, y : arrays
    cv : int | CV splitter
    scoring : 'accuracy' | 'r2' | 'neg_mse' | 'f1' | 'roc_auc'
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if isinstance(cv, int):
        splitter = StratifiedKFold(n_splits=cv) if _is_classifier(estimator) else KFold(n_splits=cv)
    else:
        splitter = cv

    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        est = copy.deepcopy(estimator)
        est.fit(X[train_idx], y[train_idx])
        score = _compute_score(est, X[val_idx], y[val_idx], scoring)
        scores.append(score)
    return np.array(scores)


def cross_validate(estimator, X: NDArray, y: NDArray, cv=5,
                   scoring: list[str] | None = None) -> dict[str, NDArray]:
    """Like cross_val_score but returns dict with multiple metrics and fit times."""
    import time
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    scoring = scoring or ["accuracy" if _is_classifier(estimator) else "r2"]
    if isinstance(cv, int):
        splitter = StratifiedKFold(n_splits=cv) if _is_classifier(estimator) else KFold(n_splits=cv)
    else:
        splitter = cv

    results: dict[str, list] = {f"test_{s}": [] for s in scoring}
    results["fit_time"] = []
    results["score_time"] = []

    for train_idx, val_idx in splitter.split(X, y):
        est = copy.deepcopy(estimator)
        t0 = time.perf_counter()
        est.fit(X[train_idx], y[train_idx])
        results["fit_time"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for s in scoring:
            results[f"test_{s}"].append(_compute_score(est, X[val_idx], y[val_idx], s))
        results["score_time"].append(time.perf_counter() - t0)

    return {k: np.array(v) for k, v in results.items()}


def _compute_score(estimator, X_val, y_val, scoring: str) -> float:
    if scoring == "accuracy":
        return float(np.mean(estimator.predict(X_val) == y_val))
    if scoring == "r2":
        return estimator.score(X_val, y_val)
    if scoring == "neg_mse":
        return -float(np.mean((estimator.predict(X_val) - y_val) ** 2))
    if scoring == "f1":
        from .metrics_classification import f1_score
        return f1_score(y_val, estimator.predict(X_val))
    if scoring == "roc_auc":
        from .metrics_classification import roc_auc_score
        proba = estimator.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)
    return estimator.score(X_val, y_val)


def _is_classifier(estimator) -> bool:
    return hasattr(estimator, "classes_") or hasattr(estimator, "predict_proba") or \
           getattr(estimator, "task", "") == "classification"


class GridSearchCV:
    """
    Exhaustive grid search over specified parameter values.

    Parameters
    ----------
    estimator : base estimator
    param_grid : dict  {'param': [val1, val2, ...]}
    cv : int | splitter
    scoring : str
    refit : bool   Refit best estimator on full training data
    """

    def __init__(self, estimator, param_grid: dict, cv=5,
                 scoring: str = "accuracy", refit: bool = True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

    def _param_combinations(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        from itertools import product
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def fit(self, X: NDArray, y: NDArray) -> "GridSearchCV":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.cv_results_: list[dict] = []
        best_score = -np.inf

        for params in self._param_combinations():
            est = copy.deepcopy(self.estimator)
            for k, v in params.items():
                setattr(est, k, v)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = scores.mean()
            self.cv_results_.append({
                "params": params,
                "mean_test_score": mean_score,
                "std_test_score": scores.std(),
                "scores": scores,
            })
            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = mean_score

        if self.refit:
            self.best_estimator_ = copy.deepcopy(self.estimator)
            for k, v in self.best_params_.items():
                setattr(self.best_estimator_, k, v)
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X: NDArray) -> NDArray:
        return self.best_estimator_.predict(X)

    def score(self, X: NDArray, y: NDArray) -> float:
        return _compute_score(self.best_estimator_, X, y, self.scoring)
