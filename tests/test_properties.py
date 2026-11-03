"""Property-based and edge-case tests for the end-to-end ``cluster`` function."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import sparse

from mutualvote import cluster
from mutualvote.core import merge_clusters


def canonical(labels: np.ndarray) -> list[int]:
    """Relabel by order of first appearance so partitions can be compared."""
    seen: dict[int, int] = {}
    return [seen.setdefault(int(v), len(seen)) for v in labels]


def blobs(seed: int, n: int, dim: int) -> np.ndarray:
    return np.random.default_rng(seed).random((n, dim)) * 3.0


GENERIC = dict(
    seed=st.integers(0, 10_000),
    n=st.integers(2, 30),
    dim=st.integers(1, 5),
)


@settings(max_examples=60, deadline=None)
@given(**GENERIC)
def test_labels_cover_every_point_and_are_contiguous(seed: int, n: int, dim: int) -> None:
    labels = cluster(blobs(seed, n, dim), metric="euclidean")
    assert labels.shape == (n,)
    assert set(labels.tolist()) == set(range(labels.max() + 1))


@settings(max_examples=40, deadline=None)
@given(**GENERIC)
def test_deterministic(seed: int, n: int, dim: int) -> None:
    x = blobs(seed, n, dim)
    assert cluster(x, metric="euclidean").tolist() == cluster(x, metric="euclidean").tolist()


@settings(max_examples=80, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(2, 14), perm_seed=st.integers(0, 10_000))
def test_row_permutation_invariance_without_ties(seed: int, n: int, perm_seed: int) -> None:
    """Holds when no two merge candidates tie exactly (docs/decisions.md #11).

    Candidate lists with continuous random scores are tie-free; lists derived from
    distances are not, because every vector's closest neighbor scores exactly 1.
    """
    rng = np.random.default_rng(seed)
    cands = [
        {v: float(rng.random()) for v in range(n) if v != u and rng.random() < 0.5}
        for u in range(n)
    ]
    perm = np.random.default_rng(perm_seed).permutation(n)  # new index j <- old index perm[j]
    new_of_old = np.empty(n, dtype=int)
    new_of_old[perm] = np.arange(n)
    permuted = [{int(new_of_old[v]): c for v, c in cands[perm[j]].items()} for j in range(n)]
    original = canonical(merge_clusters(cands))
    restored = np.empty(n, dtype=int)
    restored[perm] = merge_clusters(permuted)
    assert canonical(restored) == original


def test_two_separated_blobs_give_at_least_two_clusters() -> None:
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(0, 0.1, (25, 2)), rng.normal(10, 0.1, (25, 2))])
    labels = cluster(x, metric="euclidean")
    assert set(labels[:25]).isdisjoint(labels[25:])


# ---- edge cases -------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2])
def test_tiny_inputs(n: int) -> None:
    labels = cluster(np.zeros((n, 3), dtype=int))
    assert labels.shape == (n,)
    assert labels.tolist() == list(range(n))  # no data at all: all singletons


def test_two_identical_nonempty_rows_merge() -> None:
    assert cluster(np.array([[1, 1, 0], [1, 1, 0]])).tolist() == [0, 0]


def test_identical_rows_share_a_cluster() -> None:
    x = np.array([[1, 1, 1, 0, 0, 0]] * 5)
    assert cluster(x).tolist() == [0] * 5


def test_duplicates_stay_together_inside_blobs() -> None:
    rng = np.random.default_rng(5)
    x = np.zeros((30, 30), dtype=int)
    x[:15, :15] = rng.random((15, 15)) < 0.6
    x[15:, 15:] = rng.random((15, 15)) < 0.6
    x[1] = x[0]  # duplicate within the first blob
    x[16] = x[15]  # and within the second
    labels = cluster(x)
    assert labels[0] == labels[1]
    assert labels[15] == labels[16]


def test_empty_rows_are_singletons_and_do_not_disturb_the_rest() -> None:
    rng = np.random.default_rng(2)
    x = np.zeros((24, 24), dtype=int)
    x[:12, :12] = rng.random((12, 12)) < 0.6
    x[12:20, 12:] = rng.random((8, 12)) < 0.6
    empty = [20, 21, 22, 23]
    with_empty = cluster(x)
    without_empty = cluster(x[:20])
    assert len(set(with_empty[empty])) == len(empty)
    assert set(with_empty[empty]).isdisjoint(with_empty[:20])
    assert canonical(with_empty[:20]) == canonical(without_empty)


def test_all_empty_rows_are_all_singletons() -> None:
    assert cluster(sparse.csr_matrix((6, 4), dtype=int)).tolist() == [0, 1, 2, 3, 4, 5]


def test_sparse_input_matches_dense() -> None:
    rng = np.random.default_rng(9)
    x = (rng.random((20, 15)) < 0.35).astype(int)
    assert cluster(sparse.csr_matrix(x)).tolist() == cluster(x).tolist()


def test_precomputed_matches_metric() -> None:
    rng = np.random.default_rng(4)
    x = rng.random((15, 3))
    d = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    assert cluster(d, metric="precomputed").tolist() == cluster(x, metric="euclidean").tolist()
