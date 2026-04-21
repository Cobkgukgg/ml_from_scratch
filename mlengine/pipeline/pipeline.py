"""
sklearn-compatible Pipeline: chain transformers + final estimator.
Supports fit, transform, fit_transform, predict, predict_proba, score.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import copy
from typing import Any


class Pipeline:
    """
    Sequential pipeline of transforms ending with an estimator.

    Parameters
    ----------
    steps : list of (name, transform/estimator) tuples
    memory : bool   Cache fitted transformers (not yet implemented)

    Examples
    --------
    >>> pipe = Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("pca",    PCA(n_components=10)),
    ...     ("clf",    LogisticRegression()),
    ... ])
    >>> pipe.fit(X_train, y_train).score(X_test, y_test)
    """

    def __init__(self, steps: list[tuple[str, Any]], memory: bool = False):
        self.steps = steps
        self.memory = memory
        self._validate_steps()

    def _validate_steps(self):
        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("All step names must be unique.")
        for name, est in self.steps[:-1]:
            if not (hasattr(est, "fit") and hasattr(est, "transform")):
                raise TypeError(
                    f"Intermediate step '{name}' must have fit() and transform(). "
                    f"Got {type(est).__name__}."
                )

    # ---- properties ---- #
    @property
    def named_steps(self) -> dict:
        return dict(self.steps)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key]

    # ---- fit / transform ---- #
    def fit(self, X: NDArray, y: NDArray | None = None, **fit_params) -> "Pipeline":
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                step.fit(Xt, y)
                Xt = step.transform(Xt)
        # Final estimator
        final_name, final_est = self.steps[-1]
        final_est.fit(Xt, y, **fit_params) if fit_params else final_est.fit(Xt, y)
        return self

    def fit_transform(self, X: NDArray, y: NDArray | None = None) -> NDArray:
        """Only valid if last step is a transformer."""
        Xt = X
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                step.fit(Xt, y)
                Xt = step.transform(Xt)
        return Xt

    def transform(self, X: NDArray) -> NDArray:
        Xt = X
        for _, step in self.steps:
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X: NDArray) -> NDArray:
        Xt = self._transform_all_but_last(X)
        return self.steps[-1][1].predict(Xt)

    def predict_proba(self, X: NDArray) -> NDArray:
        Xt = self._transform_all_but_last(X)
        return self.steps[-1][1].predict_proba(Xt)

    def predict_log_proba(self, X: NDArray) -> NDArray:
        Xt = self._transform_all_but_last(X)
        return self.steps[-1][1].predict_log_proba(Xt)

    def decision_function(self, X: NDArray) -> NDArray:
        Xt = self._transform_all_but_last(X)
        return self.steps[-1][1].decision_function(Xt)

    def score(self, X: NDArray, y: NDArray) -> float:
        Xt = self._transform_all_but_last(X)
        return self.steps[-1][1].score(Xt, y)

    def _transform_all_but_last(self, X: NDArray) -> NDArray:
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return Xt

    # ---- cloning ---- #
    def clone(self) -> "Pipeline":
        return copy.deepcopy(self)

    def set_params(self, **params) -> "Pipeline":
        """Support step__param syntax like sklearn."""
        for key, value in params.items():
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                setattr(self.named_steps[step_name], param_name, value)
            else:
                setattr(self, key, value)
        return self

    def get_params(self, deep: bool = True) -> dict:
        params: dict = {}
        for name, step in self.steps:
            params[name] = step
            if deep and hasattr(step, "get_params"):
                for k, v in step.get_params().items():
                    params[f"{name}__{k}"] = v
        return params

    def __repr__(self) -> str:
        step_str = "\n  ".join(f"({name}) {type(est).__name__}" for name, est in self.steps)
        return f"Pipeline(\n  {step_str}\n)"


def make_pipeline(*steps) -> Pipeline:
    """Convenience constructor: names are lowercased class names."""
    named = []
    counts: dict[str, int] = {}
    for step in steps:
        name = type(step).__name__.lower()
        if name in counts:
            counts[name] += 1
            name = f"{name}_{counts[name]}"
        else:
            counts[name] = 0
        named.append((name, step))
    return Pipeline(named)
