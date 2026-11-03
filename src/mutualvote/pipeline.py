"""End-to-end functional entry point: data -> distances -> candidates -> clusters."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .candidates import candidate_scores
from .core import merge_clusters
from .distances import jaccard_distances, pairwise_distances

IntArray = npt.NDArray[np.intp]


def _relabel_by_first_member(raw: IntArray) -> IntArray:
    """Renumber labels to ``0..k-1`` in order of each cluster's smallest member."""
    _, first, inverse = np.unique(raw, return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(first))
    return np.asarray(rank[inverse.ravel()], dtype=np.intp)


def cluster(
    x: Any,
    *,
    metric: str = "jaccard",
    hostility: float = 0.5,
    base_rate: float = 0.5,
) -> IntArray:
    """Cluster the rows of ``x`` with Mutual Vote and return one label per row.

    ``metric`` is ``"jaccard"`` (binary data, dense or scipy sparse), ``"precomputed"``
    or any :func:`scipy.spatial.distance.pdist` metric. With ``"jaccard"``, rows that
    contain no data carry no information about who they are close to; they are set
    aside as singleton clusters instead of being clustered (docs/decisions.md #10).
    """
    if metric != "jaccard":
        dist = pairwise_distances(x, metric)
        return merge_clusters(candidate_scores(dist), hostility, base_rate)

    m = sparse.csr_matrix(x)
    m.eliminate_zeros()
    n = m.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.intp)
    active = np.flatnonzero(np.diff(m.indptr) > 0)
    raw = np.arange(n, dtype=np.intp) + n  # empty rows: unique labels above any real one
    if active.size:
        sub = merge_clusters(candidate_scores(jaccard_distances(m[active])), hostility, base_rate)
        raw[active] = sub
    return _relabel_by_first_member(raw)
