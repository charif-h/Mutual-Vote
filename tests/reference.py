"""Slow, literal transcription of Algorithm 1 used to cross-check the fast implementation.

Everything is recomputed from scratch from the member pairs on every iteration
(docs/decisions.md #4); it shares no code with ``mutualvote.core`` or ``Opinion``.
"""

from __future__ import annotations

Cands = list[dict[int, float]]
Cluster = tuple[int, ...]


def opinion_score(
    cands: Cands, src: Cluster, dst: Cluster, hostility: float = 0.5, a: float = 0.5
) -> float:
    """Shifted score of the opinion of cluster ``src`` about cluster ``dst``."""
    r = s = 0.0
    w = 2.0
    for x in src:
        for y in dst:
            c = cands[x].get(y)
            if c is None:
                s += hostility
                w += 1.0 - hostility
            else:
                r += c
                w += 1.0 - c
    total = r + s + w
    return 2.0 * (r / total + a * w / total) - 1.0


def is_linked(cands: Cands, x: Cluster, y: Cluster) -> bool:
    return any(v in cands[u] for u in x for v in y) or any(u in cands[v] for u in x for v in y)


def coupling_list(
    cands: Cands, clusters: dict[int, Cluster], hostility: float = 0.5, a: float = 0.5
) -> list[tuple[float, int, int, float, float]]:
    """Linked pairs as ``(avg, id_x, id_y, score_xy, score_yx)``, best first, ties by ids."""
    ids = sorted(clusters)
    entries = []
    for i, p in enumerate(ids):
        for q in ids[i + 1 :]:
            if not is_linked(cands, clusters[p], clusters[q]):
                continue
            sxy = opinion_score(cands, clusters[p], clusters[q], hostility, a)
            syx = opinion_score(cands, clusters[q], clusters[p], hostility, a)
            entries.append(((sxy + syx) / 2.0, p, q, sxy, syx))
    entries.sort(key=lambda e: (-e[0], e[1], e[2]))
    return entries


def reference_clusters(
    cands: Cands, hostility: float = 0.5, a: float = 0.5
) -> tuple[list[int], list[tuple[Cluster, float]]]:
    """Run the algorithm; return ``(labels, history)``.

    ``history`` lists each merge as ``(merged members, avg score)``. Labels are numbered
    by smallest member, like ``mutualvote.merge_clusters``.
    """
    clusters: dict[int, Cluster] = {i: (i,) for i in range(len(cands))}
    next_id = len(cands)
    history: list[tuple[Cluster, float]] = []
    while True:
        entries = [e for e in coupling_list(cands, clusters, hostility, a) if e[0] >= 0.0]
        if not entries:
            break
        avg, p, q, _, _ = entries[0]
        merged = tuple(sorted(clusters.pop(p) + clusters.pop(q)))
        clusters[next_id] = merged
        next_id += 1
        history.append((merged, avg))
    labels = [0] * len(cands)
    for label, members in enumerate(sorted(clusters.values(), key=min)):
        for m in members:
            labels[m] = label
    return labels, history
