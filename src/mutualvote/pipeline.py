"""End-to-end functional entry point: data -> distances -> candidates -> clusters."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse

from ._progress import say
from .candidates import candidates_from_blocks
from .core import merge_clusters
from .distances import distance_blocks

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
    chunk_size: int = 1024,
    verbose: bool = False,
) -> IntArray:
    """Cluster the rows of ``x`` with Mutual Vote and return one label per row.

    ``metric`` is ``"jaccard"`` (binary data, dense or scipy sparse), ``"precomputed"``
    or any :func:`scipy.spatial.distance.cdist` metric. With ``"jaccard"``, rows that
    contain no data carry no information about who they are close to; they are set
    aside as singleton clusters instead of being clustered (docs/decisions.md #10).

    Distances are computed ``chunk_size`` rows at a time, so the full distance matrix is
    never held in memory (except for ``"precomputed"``, where the caller already has it).
    See docs/scaling.md for time and memory at 1k-10k vectors.

    With ``verbose=True`` progress is reported on stderr for each stage (distances and
    candidates, linking, merging), with ``tqdm`` if installed. The result is unchanged.
    """
    if metric != "jaccard":
        n_rows = int(x.shape[0]) if hasattr(x, "shape") else len(x)
        say(f"{n_rows:,} vectors, metric={metric!r}", verbose)
        blocks = distance_blocks(x, metric, chunk_size)
        cands = candidates_from_blocks(blocks, n_rows=n_rows, verbose=verbose)
        return merge_clusters(cands, hostility, base_rate, verbose)

    m = sparse.csr_matrix(x)
    m.eliminate_zeros()
    n = m.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.intp)
    active = np.flatnonzero(np.diff(m.indptr) > 0)
    say(
        f"{n:,} vectors, {m.shape[1]:,} features, {m.nnz:,} non-zeros "
        f"(any non-zero value counts as 1), {n - active.size:,} empty rows set aside",
        verbose,
    )
    raw = np.arange(n, dtype=np.intp) + n  # empty rows: unique labels above any real one
    if active.size:
        blocks = distance_blocks(m[active], metric, chunk_size)
        cands = candidates_from_blocks(blocks, n_rows=int(active.size), verbose=verbose)
        raw[active] = merge_clusters(cands, hostility, base_rate, verbose)
    return _relabel_by_first_member(raw)
