"""Optional scikit-learn integration.

When scikit-learn is installed, :class:`BaseEstimator` and :class:`ClusterMixin` are the
real ones, so estimators work with ``clone``, ``Pipeline``, grid search and so on. When it
is not, minimal stand-ins with the same ``get_params`` / ``set_params`` / ``fit_predict``
behaviour are used, so the library has no hard scikit-learn dependency.
"""

from __future__ import annotations

import inspect
from typing import Any


class _FallbackBaseEstimator:
    """Minimal ``get_params`` / ``set_params`` (constructor arguments only)."""

    @classmethod
    def _get_param_names(cls) -> list[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        params = inspect.signature(init).parameters.values()
        return sorted(p.name for p in params if p.name != "self" and p.kind != p.VAR_KEYWORD)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> _FallbackBaseEstimator:
        valid = self._get_param_names()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class _FallbackClusterMixin:
    _estimator_type = "clusterer"

    def fit_predict(self, X: Any, y: Any = None, **kwargs: Any) -> Any:  # noqa: N803
        self.fit(X, **kwargs)  # type: ignore[attr-defined]
        return self.labels_  # type: ignore[attr-defined]


try:
    from sklearn.base import BaseEstimator, ClusterMixin

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only without scikit-learn
    BaseEstimator = _FallbackBaseEstimator
    ClusterMixin = _FallbackClusterMixin
    HAS_SKLEARN = False

__all__ = [
    "HAS_SKLEARN",
    "BaseEstimator",
    "ClusterMixin",
    "_FallbackBaseEstimator",
    "_FallbackClusterMixin",
]
