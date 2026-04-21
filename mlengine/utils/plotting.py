"""Visualization helpers (optional matplotlib dependency)."""

from __future__ import annotations
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")


def plot_decision_boundary(model, X, y, ax=None, resolution=200, title="Decision Boundary"):
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=30)
    ax.set_title(title)
    return ax


def plot_learning_curve(train_losses, val_losses=None, ax=None, title="Learning Curve"):
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train Loss")
    if val_losses is not None:
        ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_confusion_matrix(cm, class_names=None, ax=None, title="Confusion Matrix"):
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    if class_names is not None:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    return ax


def plot_roc_curve(fpr, tpr, auc=None, ax=None):
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    label = f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC"
    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax
