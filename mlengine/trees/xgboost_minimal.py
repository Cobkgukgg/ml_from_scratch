"""
Minimal XGBoost-style boosting with second-order Taylor expansion.

Key differences from vanilla GBDT:
  - Uses both gradient (g) and hessian (h) to compute leaf weights
  - Leaf weight: w* = -G / (H + lambda)
  - Split gain: 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
  - L2 regularization on leaf weights (lambda)
  - Min-child-weight pruning (hessian sum)
  - Column (feature) subsampling per tree
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
from ..utils.math_utils import sigmoid


@dataclass
class XGBNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["XGBNode"] = None
    right: Optional["XGBNode"] = None
    weight: float = 0.0         # leaf prediction
    gain: float = 0.0
    n_samples: int = 0


class XGBTree:
    """Single regression tree using XGBoost split criterion."""

    def __init__(self, max_depth: int = 6, min_child_weight: float = 1.0,
                 reg_lambda: float = 1.0, gamma: float = 0.0,
                 colsample: float = 1.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.colsample = colsample

    def _leaf_weight(self, g: NDArray, h: NDArray) -> float:
        return float(-g.sum() / (h.sum() + self.reg_lambda))

    def _score(self, g: NDArray, h: NDArray) -> float:
        return float(g.sum() ** 2 / (h.sum() + self.reg_lambda))

    def _best_split(self, X, g, h):
        n, p = X.shape
        best_gain, best_feat, best_thresh = -np.inf, None, None
        parent_score = self._score(g, h)
        n_feats = max(1, int(p * self.colsample))
        feat_idx = np.random.choice(p, size=n_feats, replace=False)

        for j in feat_idx:
            order = np.argsort(X[:, j])
            Xj, gj, hj = X[order, j], g[order], h[order]
            G_left, H_left = 0.0, 0.0
            G_total, H_total = g.sum(), h.sum()
            for i in range(n - 1):
                G_left += gj[i]; H_left += hj[i]
                if Xj[i] == Xj[i + 1]:
                    continue
                G_right = G_total - G_left
                H_right = H_total - H_left
                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue
                gain = 0.5 * (
                    G_left**2 / (H_left + self.reg_lambda) +
                    G_right**2 / (H_right + self.reg_lambda) -
                    parent_score
                ) - self.gamma
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = (Xj[i] + Xj[i + 1]) / 2
        return best_feat, best_thresh, best_gain

    def _build(self, X, g, h, depth):
        node = XGBNode(n_samples=len(g), weight=self._leaf_weight(g, h))
        if depth >= self.max_depth or len(g) <= 1:
            return node
        feat, thresh, gain = self._best_split(X, g, h)
        if feat is None or gain <= 0:
            return node
        mask = X[:, feat] <= thresh
        node.feature = feat
        node.threshold = thresh
        node.gain = gain
        node.left = self._build(X[mask], g[mask], h[mask], depth + 1)
        node.right = self._build(X[~mask], g[~mask], h[~mask], depth + 1)
        return node

    def fit(self, X, g, h):
        self.root_ = self._build(X, g, h, 0)
        return self

    def predict(self, X):
        def _pred(x, node):
            if node.feature is None:
                return node.weight
            return _pred(x, node.left) if x[node.feature] <= node.threshold else _pred(x, node.right)
        return np.array([_pred(x, self.root_) for x in X])


class XGBoostClassifier:
    """
    Minimal XGBoost binary classifier.

    Uses log-loss with second-order gradient (hessian = p*(1-p)).
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, min_child_weight: float = 1.0,
                 reg_lambda: float = 1.0, gamma: float = 0.0,
                 subsample: float = 1.0, colsample: float = 1.0,
                 random_state: int | None = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.random_state = random_state

    def fit(self, X: NDArray, y: NDArray) -> "XGBoostClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        p0 = np.clip(y.mean(), 1e-7, 1 - 1e-7)
        self.base_score_ = float(np.log(p0 / (1 - p0)))
        F = np.full(n, self.base_score_)
        self.trees_: list[XGBTree] = []

        for _ in range(self.n_estimators):
            probs = sigmoid(F)
            g = probs - y                 # first derivative of log-loss
            h = probs * (1 - probs)       # second derivative (hessian)
            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(n * self.subsample), replace=False)
                Xs, gs, hs = X[idx], g[idx], h[idx]
            else:
                Xs, gs, hs = X, g, h
            tree = XGBTree(max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                           reg_lambda=self.reg_lambda, gamma=self.gamma,
                           colsample=self.colsample)
            tree.fit(Xs, gs, hs)
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.base_score_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        p = sigmoid(F)
        return np.column_stack([1 - p, p])

    def predict(self, X: NDArray) -> NDArray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
