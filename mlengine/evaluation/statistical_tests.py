"""
Statistical hypothesis tests for model comparison:
  - paired t-test (5x2 CV)
  - Wilcoxon signed-rank test
  - McNemar's test
  - Friedman test + Nemenyi post-hoc
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy import stats


def paired_ttest(scores_a: NDArray, scores_b: NDArray,
                 alpha: float = 0.05) -> dict:
    """
    Paired t-test for comparing two cross-validation score arrays.
    H₀: mean(A) == mean(B).
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    diff = scores_a - scores_b
    n = len(diff)
    t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(n) + 1e-15)
    # Two-tailed p-value from t-distribution
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    return {
        "statistic": float(t_stat),
        "p_value": p_value,
        "significant": p_value < alpha,
        "mean_diff": float(diff.mean()),
        "conclusion": (
            f"A is significantly {'better' if diff.mean() > 0 else 'worse'} than B (p={p_value:.4f})"
            if p_value < alpha
            else f"No significant difference (p={p_value:.4f})"
        ),
    }


def fivex2cv_ttest(estimator_a, estimator_b, X: NDArray, y: NDArray,
                   alpha: float = 0.05, random_state: int | None = None) -> dict:
    """
    Dietterich's 5×2 CV paired t-test.
    More conservative than standard CV t-test.
    """
    import copy
    from .cross_validation import StratifiedKFold
    rng = np.random.default_rng(random_state)
    diffs = []
    for i in range(5):
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(rng.integers(1e6)))
        for train_idx, val_idx in cv.split(X, y):
            ea = copy.deepcopy(estimator_a)
            eb = copy.deepcopy(estimator_b)
            ea.fit(X[train_idx], y[train_idx])
            eb.fit(X[train_idx], y[train_idx])
            sa = np.mean(ea.predict(X[val_idx]) == y[val_idx])
            sb = np.mean(eb.predict(X[val_idx]) == y[val_idx])
            diffs.append(sa - sb)

    diffs = np.array(diffs)
    p1 = diffs[:5]
    p2 = diffs[5:]
    means = (p1 + p2) / 2
    vars_ = (p1 - means) ** 2 + (p2 - means) ** 2
    t_stat = p1[0] / np.sqrt(vars_.sum() / 5 + 1e-15)
    p_value = float(2 * stats.t.sf(abs(t_stat), df=5))
    return {
        "statistic": float(t_stat),
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def wilcoxon_test(scores_a: NDArray, scores_b: NDArray,
                  alpha: float = 0.05) -> dict:
    """
    Wilcoxon signed-rank test — non-parametric alternative to paired t-test.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    result = stats.wilcoxon(scores_a, scores_b)
    p_value = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "significant": p_value < alpha,
        "conclusion": (
            "Significant difference detected."
            if p_value < alpha
            else "No significant difference."
        ),
    }


def mcnemar_test(y_true: NDArray, y_pred_a: NDArray,
                 y_pred_b: NDArray, alpha: float = 0.05) -> dict:
    """
    McNemar's test: compare two classifiers on the same test set.
    Examines the contingency table of disagreements.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = np.sum(correct_a & ~correct_b)   # A correct, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B correct
    # Edwards' correction for continuity
    chi2 = (abs(b - c) - 1) ** 2 / (b + c + 1e-10) if (b + c) > 0 else 0.0
    p_value = float(stats.chi2.sf(chi2, df=1))
    return {
        "statistic": float(chi2),
        "p_value": p_value,
        "significant": p_value < alpha,
        "b": int(b),
        "c": int(c),
        "conclusion": (
            f"{'A' if b > c else 'B'} is significantly better (p={p_value:.4f})."
            if p_value < alpha
            else f"No significant difference (p={p_value:.4f})."
        ),
    }


def friedman_test(*score_arrays: NDArray, alpha: float = 0.05) -> dict:
    """
    Friedman test: non-parametric k-algorithm comparison across multiple datasets.
    """
    data = np.array(score_arrays).T   # shape (n_datasets, n_algorithms)
    n, k = data.shape
    # Rank within each row
    from scipy.stats import rankdata
    ranks = np.apply_along_axis(lambda row: rankdata(-row), axis=1, arr=data)
    Rj = ranks.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * np.sum((Rj - (k + 1) / 2) ** 2)
    p_value = float(stats.chi2.sf(chi2, df=k - 1))
    return {
        "statistic": float(chi2),
        "p_value": p_value,
        "significant": p_value < alpha,
        "mean_ranks": Rj,
    }
