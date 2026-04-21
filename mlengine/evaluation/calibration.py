"""Probability calibration: Platt scaling, isotonic regression, reliability diagrams."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class PlattScaler:
    """
    Platt scaling — calibrate classifier probabilities via logistic regression
    on the decision function scores.

    Reference: Platt (1999) "Probabilistic Outputs for Support Vector Machines"
    """

    def __init__(self):
        self.A_: float = 0.0
        self.B_: float = 0.0

    def fit(self, scores: NDArray, y: NDArray) -> "PlattScaler":
        """
        Fit A, B such that P(y=1|f) = 1 / (1 + exp(A*f + B)).
        Uses the modified targets from Platt's paper to avoid over-confident calibration.
        """
        scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        # Smooth targets
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1.0 / (n_neg + 2)
        t = np.where(y == 1, t_pos, t_neg)

        # Minimise log-loss via Newton's method
        A, B = 0.0, np.log((n_neg + 1) / (n_pos + 1))
        for _ in range(100):
            fval = A * scores + B
            p = 1.0 / (1.0 + np.exp(-np.clip(fval, -500, 500)))
            dA = np.dot(scores, p - t)
            dB = np.sum(p - t)
            d2A = np.dot(scores ** 2, p * (1 - p))
            d2B = np.sum(p * (1 - p))
            d2AB = np.dot(scores, p * (1 - p))
            det = d2A * d2B - d2AB ** 2 + 1e-10
            A -= (d2B * dA - d2AB * dB) / det
            B -= (d2A * dB - d2AB * dA) / det
        self.A_ = A
        self.B_ = B
        return self

    def predict_proba(self, scores: NDArray) -> NDArray:
        scores = np.asarray(scores, dtype=float)
        p = 1.0 / (1.0 + np.exp(np.clip(self.A_ * scores + self.B_, -500, 500)))
        return np.column_stack([1 - p, p])


class IsotonicCalibrator:
    """
    Isotonic regression calibrator — fits a non-decreasing step function.
    Uses the Pool Adjacent Violators (PAV) algorithm.
    """

    def fit(self, scores: NDArray, y: NDArray) -> "IsotonicCalibrator":
        scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)
        order = np.argsort(scores)
        self._x = scores[order]
        y_sorted = y[order]

        # PAV algorithm
        blocks = [[y_sorted[0], 1]]
        for yi in y_sorted[1:]:
            blocks.append([yi, 1])
            while len(blocks) >= 2 and blocks[-2][0] >= blocks[-1][0]:
                w1, n1 = blocks[-2]
                w2, n2 = blocks[-1]
                blocks[-2:] = [[(w1 * n1 + w2 * n2) / (n1 + n2), n1 + n2]]
        # Expand blocks to per-sample fitted values
        self._y_iso = np.repeat([b[0] for b in blocks], [b[1] for b in blocks])
        return self

    def predict_proba(self, scores: NDArray) -> NDArray:
        scores = np.asarray(scores, dtype=float)
        p = np.interp(scores, self._x, self._y_iso)
        return np.column_stack([1 - p, p])


class CalibratedClassifier:
    """
    Wraps any classifier and calibrates its output probabilities.

    Parameters
    ----------
    base_estimator : estimator with decision_function or predict_proba
    method : 'platt' | 'isotonic'
    cv : int   Folds for cross-validation calibration
    """

    def __init__(self, base_estimator, method: str = "platt", cv: int = 5):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X: NDArray, y: NDArray) -> "CalibratedClassifier":
        import copy
        from .cross_validation import StratifiedKFold
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        splitter = StratifiedKFold(n_splits=self.cv)
        all_scores, all_labels = [], []

        for train_idx, val_idx in splitter.split(X, y):
            est = copy.deepcopy(self.base_estimator)
            est.fit(X[train_idx], y[train_idx])
            if hasattr(est, "decision_function"):
                scores = est.decision_function(X[val_idx])
            else:
                scores = est.predict_proba(X[val_idx])[:, 1]
            all_scores.append(scores)
            all_labels.append(y[val_idx])

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        if self.method == "platt":
            self.calibrator_ = PlattScaler().fit(all_scores, (all_labels == self.classes_[1]).astype(float))
        else:
            self.calibrator_ = IsotonicCalibrator().fit(all_scores, (all_labels == self.classes_[1]).astype(float))

        # Refit base on full data
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        if hasattr(self.base_estimator, "decision_function"):
            scores = self.base_estimator.decision_function(X)
        else:
            scores = self.base_estimator.predict_proba(X)[:, 1]
        return self.calibrator_.predict_proba(scores)

    def predict(self, X: NDArray) -> NDArray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def calibration_curve(y_true: NDArray, y_prob: NDArray, n_bins: int = 10):
    """
    Compute fraction of positives vs mean predicted probability per bin.
    Returns (fraction_of_positives, mean_predicted_probability).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    frac_pos, mean_pred = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            frac_pos.append(y_true[mask].mean())
            mean_pred.append(y_prob[mask].mean())
    return np.array(frac_pos), np.array(mean_pred)


def expected_calibration_error(y_true: NDArray, y_prob: NDArray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
    return float(ece)
