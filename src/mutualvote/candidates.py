"""Candidate scores, paper equation (1)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from .neighbors import TOLERANCE, row_neighbors, row_threshold

FloatArray = npt.NDArray[np.float64]


def row_candidates(row: FloatArray, u: int) -> dict[int, float]:
    """Candidates ``{neighbor: score}`` of vector ``u`` from its distance row.

    ``score = 1 - (d - d_min) / (theta - d_min)``, in ``[0, 1]``; the closest
    neighbor scores 1. If ``theta == d_min`` (all neighbors equally close) every
    neighbor scores 1 (docs/decisions.md #6).
    """
    theta = row_threshold(row, u)
    nb = row_neighbors(row, u, theta)
    if nb.size == 0:
        return {}
    d = row[nb]
    d_min = float(d[0])  # neighbors are sorted by increasing distance
    span = theta - d_min
    if span <= TOLERANCE:
        scores = np.ones_like(d)
    else:
        scores = np.clip(1.0 - (d - d_min) / span, 0.0, 1.0)
    return {int(v): float(c) for v, c in zip(nb, scores)}


def candidates_from_blocks(blocks: Iterable[tuple[int, FloatArray]]) -> list[dict[int, float]]:
    """Candidates from row blocks ``(first_row_index, block)`` of the distance matrix.

    Only one block is in memory at a time, so the full ``n x n`` matrix is never built.
    """
    out: list[dict[int, float]] = []
    for start, block in blocks:
        for i in range(block.shape[0]):
            out.append(row_candidates(block[i], start + i))
    return out


def candidate_scores(dist: FloatArray) -> list[dict[int, float]]:
    """Candidates of every vector from a full distance matrix."""
    return candidates_from_blocks([(0, dist)])
