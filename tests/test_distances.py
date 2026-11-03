import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

from mutualvote.distances import jaccard_distances, pairwise_distances

# Table 1 of the paper: purchases of four users over 11 items
TABLE_1 = np.array(
    [
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # A
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],  # B
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],  # C
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],  # D
    ]
)


def test_matches_scipy_on_dense_data() -> None:
    rng = np.random.default_rng(0)
    x = (rng.random((30, 12)) < 0.4).astype(int)
    x[:, 0] = 1  # no empty rows (scipy defines empty/empty as 0)
    expected = squareform(pdist(x.astype(bool), metric="jaccard"))
    np.testing.assert_allclose(jaccard_distances(x), expected)


def test_sparse_equals_dense() -> None:
    d = jaccard_distances(TABLE_1)
    np.testing.assert_allclose(jaccard_distances(sparse.csr_matrix(TABLE_1)), d)
    np.testing.assert_allclose(jaccard_distances(sparse.coo_matrix(TABLE_1)), d)


def test_table_1_ordering_from_a() -> None:
    d = jaccard_distances(TABLE_1)
    assert d[0, 1] < d[0, 2] < d[0, 3]  # B closest to A, then C, D shares nothing
    assert d[0, 3] == 1.0
    assert d[0, 1] == pytest.approx(1 - 3 / 9)  # shares 3 items, union has 9
    assert d[0, 2] == pytest.approx(1 - 2 / 7)
    assert d[1, 2] == 1.0  # B and C share nothing


def test_empty_rows_are_maximally_distant() -> None:
    x = np.array([[0, 0], [0, 0], [1, 0]])
    d = jaccard_distances(x)
    assert d[0, 1] == 1.0 and d[0, 2] == 1.0
    assert np.all(np.diag(d) == 0.0)


def test_non_binary_values_count_as_one() -> None:
    np.testing.assert_allclose(
        jaccard_distances(np.array([[5, 0, 2], [1, 1, 0]])),
        jaccard_distances(np.array([[1, 0, 1], [1, 1, 0]])),
    )


def test_pairwise_other_metrics_and_precomputed() -> None:
    x = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    d = pairwise_distances(x, metric="euclidean")
    assert d[0, 1] == pytest.approx(5.0)
    np.testing.assert_allclose(pairwise_distances(d, metric="precomputed"), d)
    assert pairwise_distances(sparse.csr_matrix(x), metric="euclidean")[0, 2] == pytest.approx(10)


def test_precomputed_validation() -> None:
    with pytest.raises(ValueError):
        pairwise_distances(np.zeros((2, 3)), metric="precomputed")
    with pytest.raises(ValueError):
        pairwise_distances(-np.ones((2, 2)), metric="precomputed")
