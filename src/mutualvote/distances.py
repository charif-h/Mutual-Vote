"""Pairwise distances between vectors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.spatial.distance import cdist, pdist, squareform

FloatArray = npt.NDArray[np.float64]


def jaccard_distances(x: Any) -> FloatArray:
    """Pairwise Jaccard distance between the rows of a binary matrix.

    Any non-zero entry counts as 1. Works on dense arrays and scipy sparse
    matrices without densifying the input. The distance between two vectors that
    are both empty (or between an empty and a non-empty one) is 1: absence of data
    never makes vectors similar (see docs/decisions.md #6).
    """
    m = sparse.csr_matrix(x)
    m.data = np.ones_like(m.data, dtype=np.float64)
    m.eliminate_zeros()
    inter = (m @ m.T).toarray().astype(np.float64)
    sizes = np.asarray(m.sum(axis=1), dtype=np.float64).ravel()
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(union > 0, 1.0 - inter / union, 1.0)
    np.fill_diagonal(dist, 0.0)
    return np.asarray(dist, dtype=np.float64)


def pairwise_distances(x: Any, metric: str = "jaccard") -> FloatArray:
    """Return the dense ``n x n`` distance matrix of the rows of ``x``.

    ``metric`` is ``"jaccard"`` (sparse-friendly, the paper's choice), ``"precomputed"``
    (``x`` is already a square distance matrix) or any metric name accepted by
    :func:`scipy.spatial.distance.pdist`.
    """
    if metric == "precomputed":
        d = np.asarray(x, dtype=np.float64)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("precomputed distances must be a square matrix")
        if np.isnan(d).any() or (d < 0).any():
            raise ValueError("precomputed distances must be non-negative and not NaN")
        return d
    if metric == "jaccard":
        return jaccard_distances(x)
    dense = x.toarray() if sparse.issparse(x) else np.asarray(x)
    if dense.ndim != 2:
        raise ValueError("expected a 2-D array")
    if dense.shape[0] < 2:
        return np.zeros((dense.shape[0], dense.shape[0]))
    return np.asarray(squareform(pdist(dense, metric=metric)), dtype=np.float64)


def _check_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")


def distance_blocks(
    x: Any, metric: str = "jaccard", chunk_size: int = 1024
) -> Iterator[tuple[int, FloatArray]]:
    """Yield ``(first_row, block)`` row blocks of the distance matrix, ``chunk_size`` rows each.

    Same values as :func:`pairwise_distances` but with at most ``chunk_size x n`` distances
    in memory at a time.
    """
    _check_chunk_size(chunk_size)
    if metric == "precomputed":
        d = pairwise_distances(x, metric)
        for start in range(0, d.shape[0], chunk_size):
            yield start, d[start : start + chunk_size]
        return
    if metric == "jaccard":
        m = sparse.csr_matrix(x)
        m.data = np.ones_like(m.data, dtype=np.float64)
        m.eliminate_zeros()
        mt = m.T.tocsr()
        sizes = np.asarray(m.sum(axis=1), dtype=np.float64).ravel()
        n = m.shape[0]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            inter = (m[start:stop] @ mt).toarray().astype(np.float64)
            union = sizes[start:stop, None] + sizes[None, :] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                block = np.where(union > 0, 1.0 - inter / union, 1.0)
            block[np.arange(stop - start), np.arange(start, stop)] = 0.0
            yield start, np.asarray(block, dtype=np.float64)
        return
    dense = x.toarray() if sparse.issparse(x) else np.asarray(x)
    if dense.ndim != 2:
        raise ValueError("expected a 2-D array")
    for start in range(0, dense.shape[0], chunk_size):
        yield start, np.asarray(cdist(dense[start : start + chunk_size], dense, metric=metric))
