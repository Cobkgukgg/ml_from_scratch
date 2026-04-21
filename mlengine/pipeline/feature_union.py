"""
FeatureUnion: run multiple transformers in parallel and concatenate outputs.
Useful for combining heterogeneous feature extractors.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any
import copy


class FeatureUnion:
    """
    Combine outputs of multiple transformer objects.

    Parameters
    ----------
    transformer_list : list of (name, transformer) tuples
    weights : list of floats | None   Weight each transformer's output

    Examples
    --------
    >>> union = FeatureUnion([
    ...     ("pca",    PCA(n_components=5)),
    ...     ("kbest",  SelectKBest(k=10)),
    ... ])
    >>> X_combined = union.fit_transform(X, y)
    """

    def __init__(self, transformer_list: list[tuple[str, Any]],
                 weights: list[float] | None = None):
        self.transformer_list = transformer_list
        self.weights = weights

    @property
    def named_transformers(self) -> dict:
        return dict(self.transformer_list)

    def fit(self, X: NDArray, y: NDArray | None = None) -> "FeatureUnion":
        for _, t in self.transformer_list:
            t.fit(X, y) if y is not None else t.fit(X)
        return self

    def transform(self, X: NDArray) -> NDArray:
        parts = []
        for i, (_, t) in enumerate(self.transformer_list):
            Xt = t.transform(X)
            if self.weights is not None:
                Xt = Xt * self.weights[i]
            parts.append(Xt)
        return np.hstack(parts)

    def fit_transform(self, X: NDArray, y: NDArray | None = None) -> NDArray:
        parts = []
        for i, (_, t) in enumerate(self.transformer_list):
            if hasattr(t, "fit_transform"):
                Xt = t.fit_transform(X, y) if y is not None else t.fit_transform(X)
            else:
                t.fit(X, y) if y is not None else t.fit(X)
                Xt = t.transform(X)
            if self.weights is not None:
                Xt = Xt * self.weights[i]
            parts.append(Xt)
        return np.hstack(parts)

    def get_params(self, deep: bool = True) -> dict:
        params: dict = {}
        for name, t in self.transformer_list:
            params[name] = t
            if deep and hasattr(t, "get_params"):
                for k, v in t.get_params().items():
                    params[f"{name}__{k}"] = v
        return params

    def set_params(self, **params) -> "FeatureUnion":
        for key, value in params.items():
            if "__" in key:
                name, param = key.split("__", 1)
                setattr(self.named_transformers[name], param, value)
            else:
                setattr(self, key, value)
        return self

    def clone(self) -> "FeatureUnion":
        return copy.deepcopy(self)


def make_union(*transformers, weights=None) -> FeatureUnion:
    """Convenience constructor: names are lowercased class names."""
    named = []
    counts: dict[str, int] = {}
    for t in transformers:
        name = type(t).__name__.lower()
        if name in counts:
            counts[name] += 1
            name = f"{name}_{counts[name]}"
        else:
            counts[name] = 0
        named.append((name, t))
    return FeatureUnion(named, weights=weights)
