"""The mutual-vote merging loop (paper Algorithm 1)."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.intp]

#: Non-informative prior weight of a fresh opinion (see ``mutualvote.opinion``).
_PRIOR_W = 2.0


def _validate(candidates: Sequence[dict[int, float]], hostility: float, base_rate: float) -> None:
    if not 0.0 <= hostility <= 1.0:
        raise ValueError("hostility must be in [0, 1]")
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError("base_rate must be in [0, 1]")
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
    Opinions are plain ``(r, s, w)`` tuples with the score of
    :class:`mutualvote.opinion.Opinion` computed inline, because this loop dominates the
    running time. Ties are broken by cluster id. Labels are ``0..k-1`` numbered by smallest
    member index.
    """
    _validate(candidates, hostility, base_rate)
    n = len(candidates)
    if n == 0:
        return np.empty(0, dtype=np.intp)

    lam = hostility
    a = base_rate
    one_minus_lam = 1.0 - lam
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    # op[x][y] = (r, s, w) opinion of cluster x about cluster y (present iff linked)
    op: dict[int, dict[int, tuple[float, float, float]]] = {i: {} for i in range(n)}

    def interaction(x: int, y: int) -> tuple[float, float, float]:
        c = candidates[x].get(y)
        if c is None:
            return (0.0, lam, _PRIOR_W + one_minus_lam)
        return (c, 0.0, _PRIOR_W + (1.0 - c))

    for x in range(n):
        for y in candidates[x]:
            if y not in op[x]:
                op[x][y] = interaction(x, y)
                op[y][x] = interaction(y, x)

    heap: list[tuple[float, int, int]] = []
    for x in range(n):
        ox = op[x]
        for y, (r1, s1, w1) in ox.items():
            if x < y:
                r2, s2, w2 = op[y][x]
                mean = (r1 + a * w1) / (r1 + s1 + w1) + (r2 + a * w2) / (r2 + s2 + w2) - 1.0
                if mean >= 0.0:
                    heap.append((-mean, x, y))
    heapq.heapify(heap)

    next_id = n
    while heap:
        _, x, y = heapq.heappop(heap)
        if x not in members or y not in members:
            continue  # stale entry: one of the clusters was merged away
        z = next_id
        next_id += 1
        ox, oy = op[x], op[y]
        size_x, size_y = len(members[x]), len(members[y])
        oz: dict[int, tuple[float, float, float]] = {}
        op[z] = oz
        for w in (ox.keys() | oy.keys()) - {x, y}:
            ow = op[w]
            size_w = len(members[w])
            # opinion of z about w: sum of x's and y's opinions (hostile if not linked)
            p = ox.get(w)
            if p is None:
                h = size_x * size_w
                p = (0.0, lam * h, _PRIOR_W + one_minus_lam * h)
            q = oy.get(w)
            if q is None:
                h = size_y * size_w
                q = (0.0, lam * h, _PRIOR_W + one_minus_lam * h)
            r1, s1, w1 = p[0] + q[0], p[1] + q[1], p[2] + q[2] - _PRIOR_W
            # opinion of w about z
            p = ow.pop(x, None)
            if p is None:
                h = size_x * size_w
                p = (0.0, lam * h, _PRIOR_W + one_minus_lam * h)
            q = ow.pop(y, None)
            if q is None:
                h = size_y * size_w
                q = (0.0, lam * h, _PRIOR_W + one_minus_lam * h)
            r2, s2, w2 = p[0] + q[0], p[1] + q[1], p[2] + q[2] - _PRIOR_W
            oz[w] = (r1, s1, w1)
            ow[z] = (r2, s2, w2)
            mean = (r1 + a * w1) / (r1 + s1 + w1) + (r2 + a * w2) / (r2 + s2 + w2) - 1.0
            if mean >= 0.0:
                heapq.heappush(heap, (-mean, w, z))
        big, small = (x, y) if size_x >= size_y else (y, x)
        merged = members.pop(big)
        merged.extend(members.pop(small))
        members[z] = merged
        del op[x], op[y]

    labels = np.empty(n, dtype=np.intp)
    for label, group in enumerate(sorted(members.values(), key=min)):
        labels[group] = label
    return labels
