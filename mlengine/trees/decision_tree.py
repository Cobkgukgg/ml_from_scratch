"""Decision Tree for classification and regression (CART algorithm)."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional
from ..utils.math_utils import gini_impurity


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[float] = None      # leaf: predicted value / class probabilities
    n_samples: int = 0
    impurity: float = 0.0


class DecisionTree:
    """
    CART decision tree supporting classification and regression.

    Parameters
    ----------
    task : 'classification' | 'regression'
    criterion : 'gini' | 'entropy' | 'mse'
    max_depth : int | None
    min_samples_split : int
    min_samples_leaf : int
    max_features : int | float | 'sqrt' | 'log2' | None
    """

    def __init__(self, task: str = "classification", criterion: str | None = None,
                 max_depth: int | None = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features=None):
        self.task = task
        if criterion is None:
            self.criterion = "mse" if task == "regression" else "gini"
        else:
            self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    # ------------------------------------------------------------------ #
    #  Impurity                                                            #
    # ------------------------------------------------------------------ #
    def _impurity(self, y: NDArray) -> float:
        if len(y) == 0:
            return 0.0
        if self.criterion == "mse" or self.task == "regression":
            return float(np.var(y))
        if self.criterion == "gini":
            return gini_impurity(y.astype(int))
        if self.criterion == "entropy":
            counts = np.bincount(y.astype(int))
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))
        return float(np.var(y))

    def _leaf_value(self, y: NDArray):
        if self.task == "classification":
            return int(np.bincount(y.astype(int)).argmax())
        return float(y.mean())

    # ------------------------------------------------------------------ #
    #  Split search                                                        #
    # ------------------------------------------------------------------ #
    def _best_split(self, X: NDArray, y: NDArray):
        n, p = X.shape
        best_gain, best_feat, best_thresh = -np.inf, None, None
        parent_imp = self._impurity(y)

        features = self._sample_features(p)
        for j in features:
            thresholds = np.unique(X[:, j])
            for t in thresholds[:-1]:  # skip last to ensure both sides non-empty
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                gain = parent_imp - (
                    left_mask.sum() / n * self._impurity(y[left_mask]) +
                    right_mask.sum() / n * self._impurity(y[right_mask])
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, j, t
        return best_feat, best_thresh

    def _sample_features(self, p: int) -> NDArray:
        mf = self.max_features
        if mf is None:
            return np.arange(p)
        if mf == "sqrt":
            k = max(1, int(np.sqrt(p)))
        elif mf == "log2":
            k = max(1, int(np.log2(p)))
        elif isinstance(mf, float):
            k = max(1, int(mf * p))
        else:
            k = int(mf)
        return np.random.choice(p, size=min(k, p), replace=False)

    # ------------------------------------------------------------------ #
    #  Tree construction                                                   #
    # ------------------------------------------------------------------ #
    def _build(self, X: NDArray, y: NDArray, depth: int) -> Node:
        node = Node(n_samples=len(y), impurity=self._impurity(y))
        if (self.max_depth is not None and depth >= self.max_depth) or \
                len(y) < self.min_samples_split or \
                len(np.unique(y)) == 1:
            node.value = self._leaf_value(y)
            return node

        feat, thresh = self._best_split(X, y)
        if feat is None:
            node.value = self._leaf_value(y)
            return node

        left_mask = X[:, feat] <= thresh
        node.feature = feat
        node.threshold = thresh
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: NDArray, y: NDArray) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.root_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x: NDArray, node: Node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(y_pred == y))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-15))

    @property
    def feature_importances_(self) -> NDArray:
        importances = np.zeros(self.n_features_in_)
        def _traverse(node):
            if node.value is not None:
                return
            gain = node.impurity - (
                node.left.n_samples / node.n_samples * node.left.impurity +
                node.right.n_samples / node.n_samples * node.right.impurity
            )
            importances[node.feature] += gain * node.n_samples
            _traverse(node.left)
            _traverse(node.right)
        _traverse(self.root_)
        total = importances.sum()
        return importances / (total + 1e-15)
