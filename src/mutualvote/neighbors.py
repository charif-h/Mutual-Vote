"""Per-vector neighbor selection (paper Section 3, "neighbors list")."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

#: Absolute tolerance so equal distances are not split by floating-point error.
TOLERANCE = 1e-12


def distance_thresholds(dist: FloatArray) -> FloatArray:
    """Return ``theta_u = mean(D(u)) - std(D(u))`` for every vector.

    ``D(u)`` are the distances from ``u`` to all *other* vectors; the std is the
    population std. With fewer than two vectors the result is ``nan``.
    """
    n = dist.shape[0]
    theta = np.full(n, np.nan)
    if n < 2:
        return theta
    mask = ~np.eye(n, dtype=bool)
    others = dist[mask].reshape(n, n - 1)
    theta[:] = others.mean(axis=1) - others.std(axis=1)
    return theta


def select_neighbors(dist: FloatArray) -> list[IntArray]:
    """For each vector, the indices ``v != u`` with ``d(u, v) <= theta_u``.

    Indices are sorted by increasing distance (ties by index).
    """
    n = dist.shape[0]
    if n < 2:
        return [np.empty(0, dtype=np.intp) for _ in range(n)]
    theta = distance_thresholds(dist)
    result: list[IntArray] = []
    for u in range(n):
        row = dist[u]
        idx = np.flatnonzero(row <= theta[u] + TOLERANCE)
        idx = idx[idx != u]
        order = np.lexsort((idx, row[idx]))
        result.append(idx[order].astype(np.intp))
    return result
