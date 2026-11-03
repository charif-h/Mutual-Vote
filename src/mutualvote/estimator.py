"""scikit-learn-style estimator for Mutual Vote."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._compat import BaseEstimator, ClusterMixin
from .pipeline import cluster

IntArray = npt.NDArray[np.intp]


class MutualVote(ClusterMixin, BaseEstimator):  # type: ignore[misc]
    """Mutual Vote density clustering (Haydar & Boyer, UMAP'17).

    Each vector picks the neighbors that are significantly close to it (distance below
    its own mean minus one standard deviation), scores them, and clusters then merge in
    order of mutual subjective-logic opinion until no pair has a positive mutual opinion.
    There is no global distance threshold and no number of clusters to choose.

    Parameters
    ----------
    hostility : float, default=0.5
        Hostility barometer in ``[0, 1]`` (paper's lambda): how negative a vector's
        opinion of a vector that is *not* among its candidates is. 0.5 is neutral.
    base_rate : float, default=0.5
        Subjective-logic base rate ``a`` in ``[0, 1]``.
    metric : str, default="jaccard"
        ``"jaccard"`` for binary data (dense or scipy sparse), ``"precomputed"`` if ``X`` is
        a square distance matrix, or any metric of :func:`scipy.spatial.distance.pdist`.
        With ``"jaccard"``, rows without data become singleton clusters.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label of each sample, ``0..n_clusters_-1`` numbered by each cluster's
        smallest sample index. Every sample is labelled (there is no noise label).
    n_clusters_ : int
        Number of clusters found.
    n_features_in_ : int
        Number of features seen in ``fit`` (not set for ``metric="precomputed"``).

    Examples
    --------
    >>> import numpy as np
    >>> from mutualvote import MutualVote
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    >>> MutualVote().fit_predict(X).tolist()
    [0, 0, 1, 1]
    """

    labels_: IntArray
    n_clusters_: int
    n_features_in_: int

    def __init__(
        self,
        hostility: float = 0.5,
        base_rate: float = 0.5,
        metric: str = "jaccard",
    ) -> None:
        self.hostility = hostility
        self.base_rate = base_rate
        self.metric = metric

    def _check_params(self) -> None:
        for name in ("hostility", "base_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
        if not isinstance(self.metric, str):
            raise ValueError(f"metric must be a string, got {self.metric!r}")

    def fit(self, X: Any, y: Any = None) -> MutualVote:  # noqa: N803
        """Cluster ``X`` (``n_samples x n_features``, or ``n x n`` if precomputed).

        ``y`` is ignored; it exists for scikit-learn API compatibility.
        """
        self._check_params()
        self.labels_ = cluster(
            X, metric=self.metric, hostility=self.hostility, base_rate=self.base_rate
        )
        self.n_clusters_ = int(self.labels_.max()) + 1 if self.labels_.size else 0
        if self.metric != "precomputed":
            self.n_features_in_ = int(X.shape[1]) if hasattr(X, "shape") else len(X[0])
        return self

    def fit_predict(self, X: Any, y: Any = None) -> IntArray:  # noqa: N803
        """Fit and return ``labels_``."""
        return self.fit(X).labels_
