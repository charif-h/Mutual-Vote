"""The mutual-vote merging loop (paper Algorithm 1)."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .opinion import Opinion

IntArray = npt.NDArray[np.intp]


def _validate(candidates: Sequence[dict[int, float]]) -> None:
    n = len(candidates)
    for u, cands in enumerate(candidates):
        for v, c in cands.items():
            if not 0 <= v < n or v == u:
                raise ValueError(f"invalid candidate {v} for vector {u}")
            if not 0.0 <= c <= 1.0:
                raise ValueError(f"candidate score {c} for ({u}, {v}) not in [0, 1]")


def merge_clusters(
    candidates: Sequence[dict[int, float]],
    hostility: float = 0.5,
    base_rate: float = 0.5,
) -> IntArray:
    """Cluster vectors from their candidate lists; return one label per vector.

    Every vector starts as a cluster. For each linked cluster pair ``(X, Y)`` the opinion
    of ``X`` about ``Y`` sums one interaction per member pair: positive with the candidate
    score if the member is a candidate, hostile otherwise. The pair with the highest mean
    shifted score is merged, opinions towards the merged cluster are re-aggregated, and
    this repeats until the best mean score is negative.

    Implementation: a max-heap with lazy deletion (cluster ids are never reused, so an
    entry is valid iff both of its clusters are still alive) and per-cluster adjacency
    dicts, so a merge costs time proportional to the degree of the merged clusters.
    Ties are broken by cluster id. Labels are ``0..k-1`` numbered by smallest member index.
    """
    _validate(candidates)
    n = len(candidates)
    if n == 0:
        return np.empty(0, dtype=np.intp)

    prior = Opinion(a=base_rate)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    # op[x][y] = opinion of cluster x about cluster y (present iff the pair is linked)
    op: dict[int, dict[int, Opinion]] = {i: {} for i in range(n)}

    def interaction(x: int, y: int) -> Opinion:
        c = candidates[x].get(y)
        return prior.hostile(hostility, 1) if c is None else prior.positive(c)

    for x in range(n):
        for y in candidates[x]:
            if y not in op[x]:
                op[x][y] = interaction(x, y)
                op[y][x] = interaction(y, x)

    heap: list[tuple[float, int, int]] = []
    for x in range(n):
        for y in op[x]:
            if x < y:
                s = (op[x][y].score() + op[y][x].score()) / 2.0
                if s >= 0.0:
                    heap.append((-s, x, y))
    heapq.heapify(heap)

    def about(a: int, b: int) -> Opinion:
        """Opinion of ``a`` about ``b``; unlinked pairs are fully hostile."""
        found = op[a].get(b)
        if found is not None:
            return found
        return prior.hostile(hostility, len(members[a]) * len(members[b]))

    next_id = n
    while heap:
        _, x, y = heapq.heappop(heap)
        if x not in members or y not in members:
            continue  # stale entry: one of the clusters was merged away
        z = next_id
        next_id += 1
        op[z] = {}
        for w in (op[x].keys() | op[y].keys()) - {x, y}:
            z_to_w = about(x, w).combine(about(y, w))
            w_to_z = about(w, x).combine(about(w, y))
            op[w].pop(x, None)
            op[w].pop(y, None)
            op[z][w] = z_to_w
            op[w][z] = w_to_z
            s = (z_to_w.score() + w_to_z.score()) / 2.0
            if s >= 0.0:
                heapq.heappush(heap, (-s, w, z))
        big, small = (x, y) if len(members[x]) >= len(members[y]) else (y, x)
        merged = members.pop(big)
        merged.extend(members.pop(small))
        members[z] = merged
        del op[x], op[y]

    labels = np.empty(n, dtype=np.intp)
    for label, group in enumerate(sorted(members.values(), key=min)):
        labels[group] = label
    return labels
