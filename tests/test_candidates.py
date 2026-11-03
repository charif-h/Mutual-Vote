import numpy as np
import pytest

from mutualvote.candidates import candidate_scores
from mutualvote.neighbors import distance_thresholds, select_neighbors


def test_threshold_is_mean_minus_std_of_other_distances() -> None:
    d = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    theta = distance_thresholds(d)
    assert theta[0] == pytest.approx(2.0 - 1.0)  # others: 1, 3 -> mean 2, std 1
    assert theta[1] == pytest.approx(1.5 - 0.5)


def _two_groups() -> np.ndarray:
    """8 points: {0,1,2,3} and {4,5,6,7}, within-group 0.1 (except from 0), across 0.9."""
    d = np.full((8, 8), 0.9)
    d[:4, :4] = 0.1
    d[4:, 4:] = 0.1
    d[0, 1], d[0, 2], d[0, 3] = 0.1, 0.15, 0.2
    d[1:4, 0] = d[0, 1:4]
    np.fill_diagonal(d, 0.0)
    return d


def test_neighbors_are_the_significantly_close_ones() -> None:
    nb = select_neighbors(_two_groups())
    assert nb[0].tolist() == [1, 2, 3]
    assert nb[4].tolist() == [5, 6, 7]
    assert all(u not in nb[u] for u in range(8))


def test_candidate_scores_follow_equation_1() -> None:
    d = _two_groups()
    cands = candidate_scores(d)
    theta0 = distance_thresholds(d)[0]
    assert cands[0][1] == 1.0  # closest neighbor scores 1
    assert cands[0][2] == pytest.approx(1 - (0.15 - 0.1) / (theta0 - 0.1))
    assert cands[0][3] == pytest.approx(1 - (0.2 - 0.1) / (theta0 - 0.1))
    assert cands[0][1] > cands[0][2] > cands[0][3] >= 0.0
    assert all(0.0 <= c <= 1.0 for cs in cands for c in cs.values())


def test_all_equal_distances_give_no_zero_division() -> None:
    d = np.ones((4, 4)) - np.eye(4)
    cands = candidate_scores(d)
    assert all(set(c.values()) == {1.0} for c in cands)
    assert all(len(c) == 3 for c in cands)


def test_vector_without_neighbors() -> None:
    # distances 0.5, 0.5, 0.9 -> theta below the closest distance
    d = np.array([[0, 0.5, 0.5, 0.9], [0.5, 0, 0.5, 0.5], [0.5, 0.5, 0, 0.5], [0.9, 0.5, 0.5, 0]])
    assert candidate_scores(d)[0] == {}


@pytest.mark.parametrize("n", [0, 1])
def test_tiny_inputs(n: int) -> None:
    assert candidate_scores(np.zeros((n, n))) == [{} for _ in range(n)]
