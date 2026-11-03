import numpy as np
import pytest
from scipy import sparse

from mutualvote import MutualVote, cluster
from mutualvote.candidates import candidate_scores, candidates_from_blocks, row_candidates
from mutualvote.distances import distance_blocks, jaccard_distances, pairwise_distances


def binary_data(seed: int = 0, n: int = 60, items: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, items), dtype=int)
    half = n // 2
    x[:half, : items // 2] = rng.random((half, items // 2)) < 0.5
    x[half:, items // 2 :] = rng.random((n - half, items - items // 2)) < 0.5
    x[rng.random((n, items)) < 0.03] = 1
    return x


def stitch(blocks: object) -> np.ndarray:
    return np.vstack([b for _, b in blocks])  # type: ignore[attr-defined]


@pytest.mark.parametrize("chunk", [1, 7, 60, 1000])
def test_jaccard_blocks_equal_full_matrix(chunk: int) -> None:
    x = binary_data()
    full = jaccard_distances(x)
    np.testing.assert_array_equal(stitch(distance_blocks(x, "jaccard", chunk)), full)
    sp = sparse.csr_matrix(x)
    np.testing.assert_array_equal(stitch(distance_blocks(sp, "jaccard", chunk)), full)


@pytest.mark.parametrize("metric", ["euclidean", "cityblock"])
def test_other_metric_blocks_equal_full_matrix(metric: str) -> None:
    x = np.random.default_rng(1).random((30, 4))
    full = pairwise_distances(x, metric)
    np.testing.assert_allclose(stitch(distance_blocks(x, metric, 8)), full, atol=1e-12)


def test_precomputed_blocks() -> None:
    d = pairwise_distances(np.random.default_rng(2).random((20, 3)), "euclidean")
    np.testing.assert_array_equal(stitch(distance_blocks(d, "precomputed", 6)), d)


@pytest.mark.parametrize("chunk", [1, 5, 13, 1024])
def test_chunk_size_never_changes_the_result(chunk: int) -> None:
    x = binary_data(3)
    assert cluster(x, chunk_size=chunk).tolist() == cluster(x).tolist()
    pts = np.random.default_rng(4).random((40, 3))
    expected = cluster(pts, metric="euclidean")
    assert cluster(pts, metric="euclidean", chunk_size=chunk).tolist() == expected.tolist()


def test_candidates_from_blocks_equal_candidate_scores() -> None:
    d = jaccard_distances(binary_data(5))
    assert candidates_from_blocks(distance_blocks(binary_data(5), "jaccard", 9)) == (
        candidate_scores(d)
    )


def test_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        list(distance_blocks(binary_data(), "jaccard", 0))


def test_row_candidates_matches_matrix_level() -> None:
    d = jaccard_distances(binary_data(6))
    full = candidate_scores(d)
    assert all(row_candidates(d[u], u) == full[u] for u in range(d.shape[0]))


def test_estimator_forwards_chunk_size() -> None:
    x = binary_data(9)
    est = MutualVote(chunk_size=11).fit(x)
    assert est.labels_.tolist() == cluster(x).tolist()
