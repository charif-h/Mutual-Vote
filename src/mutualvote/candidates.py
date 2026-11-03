"""Candidate scores, paper equation (1)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .neighbors import TOLERANCE, distance_thresholds, select_neighbors

FloatArray = npt.NDArray[np.float64]


def candidate_scores(dist: FloatArray) -> list[dict[int, float]]:
    """Return, for each vector ``u``, its candidates ``{neighbor: score}``.

    ``score = 1 - (d - d_min) / (theta - d_min)``, in ``[0, 1]``; the closest
    neighbor scores 1. If ``theta == d_min`` (all neighbors equally close) every
    neighbor scores 1 (docs/decisions.md #6).
    """
    n = dist.shape[0]
    theta = distance_thresholds(dist)
    neighbors = select_neighbors(dist)
    out: list[dict[int, float]] = []
    for u in range(n):
        nb = neighbors[u]
        if nb.size == 0:
            out.append({})
            continue
        d = dist[u, nb]
        d_min = float(d.min())
        span = float(theta[u]) - d_min
        if span <= TOLERANCE:
            scores = np.ones_like(d)
        else:
            scores = np.clip(1.0 - (d - d_min) / span, 0.0, 1.0)
        out.append({int(v): float(c) for v, c in zip(nb, scores)})
    return out
