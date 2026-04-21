"""Classification metrics: accuracy, precision, recall, F1, AUC-ROC, log-loss, MCC."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def accuracy_score(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: NDArray, y_pred: NDArray) -> NDArray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[c2i[t], c2i[p]] += 1
    return cm


def precision_score(y_true: NDArray, y_pred: NDArray, average: str = "binary",
                    zero_division: float = 0.0) -> float | NDArray:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    prec = tp / (tp + fp + 1e-15)
    prec = np.where((tp + fp) == 0, zero_division, prec)
    return _aggregate(prec, y_true, y_pred, average)


def recall_score(y_true: NDArray, y_pred: NDArray, average: str = "binary",
                 zero_division: float = 0.0) -> float | NDArray:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    rec = tp / (tp + fn + 1e-15)
    rec = np.where((tp + fn) == 0, zero_division, rec)
    return _aggregate(rec, y_true, y_pred, average)


def f1_score(y_true: NDArray, y_pred: NDArray, average: str = "binary",
             zero_division: float = 0.0) -> float | NDArray:
    p = precision_score(y_true, y_pred, average="none", zero_division=zero_division)
    r = recall_score(y_true, y_pred, average="none", zero_division=zero_division)
    f1 = 2 * p * r / (p + r + 1e-15)
    f1 = np.where((p + r) == 0, zero_division, f1)
    return _aggregate(f1, y_true, y_pred, average)


def _aggregate(scores: NDArray, y_true, y_pred, average: str):
    classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    if average == "none":
        return scores
    if average == "micro":
        return float(accuracy_score(y_true, y_pred))
    if average == "macro":
        return float(scores.mean())
    if average == "weighted":
        weights = np.array([(np.asarray(y_true) == c).sum() for c in classes])
        return float(np.average(scores, weights=weights))
    # binary: return second class
    return float(scores[-1])


def roc_auc_score(y_true: NDArray, y_score: NDArray) -> float:
    """Binary ROC AUC via trapezoidal rule."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc = np.argsort(-y_score)
    y_true_sorted = y_true[desc]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tp_rate = tp / (tp[-1] + 1e-15)
    fp_rate = fp / (fp[-1] + 1e-15)
    return float(np.trapezoid(tp_rate, fp_rate)) if hasattr(np, "trapezoid") else float(np.trapz(tp_rate, fp_rate))


def roc_curve(y_true: NDArray, y_score: NDArray):
    """Returns (fpr, tpr, thresholds)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    desc = np.argsort(-y_score)
    y_true_s = y_true[desc]
    thresholds = y_score[desc]
    tp = np.cumsum(y_true_s)
    fp = np.cumsum(1 - y_true_s)
    tpr = np.concatenate([[0], tp / (tp[-1] + 1e-15)])
    fpr = np.concatenate([[0], fp / (fp[-1] + 1e-15)])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    return fpr, tpr, thresholds


def precision_recall_curve(y_true: NDArray, y_score: NDArray):
    """Returns (precision, recall, thresholds)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    desc = np.argsort(-y_score)
    y_true_s = y_true[desc]
    thresholds = y_score[desc]
    tp = np.cumsum(y_true_s)
    fp = np.cumsum(1 - y_true_s)
    precision = tp / (tp + fp + 1e-15)
    recall = tp / (y_true.sum() + 1e-15)
    return (
        np.concatenate([precision, [1.0]]),
        np.concatenate([recall, [0.0]]),
        thresholds,
    )


def log_loss(y_true: NDArray, y_prob: NDArray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), eps, 1 - eps)
    if y_prob.ndim == 1:
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
    y_enc = np.eye(y_prob.shape[1])[y_true]
    return float(-np.mean(np.sum(y_enc * np.log(y_prob), axis=1)))


def matthews_corrcoef(y_true: NDArray, y_pred: NDArray) -> float:
    """Matthews Correlation Coefficient (binary)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / (denom + 1e-15))


def classification_report(y_true: NDArray, y_pred: NDArray,
                           target_names: list | None = None) -> str:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    header = f"{'':>12} {'precision':>10} {'recall':>8} {'f1-score':>10} {'support':>9}\n\n"
    lines = [header]
    for i, c in enumerate(classes):
        name = (target_names[i] if target_names else str(c))[:12]
        p = precision_score(y_true, y_pred, average="none")[i]
        r = recall_score(y_true, y_pred, average="none")[i]
        f = f1_score(y_true, y_pred, average="none")[i]
        s = (y_true == c).sum()
        lines.append(f"{name:>12} {p:>10.2f} {r:>8.2f} {f:>10.2f} {s:>9}\n")
    acc = accuracy_score(y_true, y_pred)
    n = len(y_true)
    lines.append(f"\n{'accuracy':>12} {'':>10} {'':>8} {acc:>10.2f} {n:>9}\n")
    return "".join(lines)
