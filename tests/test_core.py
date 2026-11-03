import numpy as np
import pytest

from mutualvote.candidates import candidate_scores
from mutualvote.core import merge_clusters
from mutualvote.distances import jaccard_distances

# Paper worked example: A=0, B=1, C=2, D=3 (candidate scores per decisions.md #1, #8)
A, B, C, D = range(4)
PAPER = [
    {B: 1.0, C: 0.8},  # A
    {A: 1.0, D: 0.9, C: 0.9},  # B
    {A: 0.6, B: 0.7},  # C
    {B: 0.7},  # D
]


def test_paper_example_ends_with_abc_and_d() -> None:
    assert merge_clusters(PAPER).tolist() == [0, 0, 0, 1]


def test_empty_and_single() -> None:
    assert merge_clusters([]).tolist() == []
    assert merge_clusters([{}]).tolist() == [0]


def test_no_candidates_gives_singletons() -> None:
    assert merge_clusters([{}, {}, {}]).tolist() == [0, 1, 2]


def test_mutual_pair_merges() -> None:
    assert merge_clusters([{1: 1.0}, {0: 1.0}]).tolist() == [0, 0]


def test_weak_one_sided_pair_does_not_merge() -> None:
    # c < 0.5 and only one side: average shifted score is negative
    assert merge_clusters([{1: 0.2}, {}]).tolist() == [0, 1]


def test_strong_one_sided_pair_merges() -> None:
    assert merge_clusters([{1: 1.0}, {}]).tolist() == [0, 0]


def test_invalid_candidates_rejected() -> None:
    with pytest.raises(ValueError):
        merge_clusters([{5: 1.0}])
    with pytest.raises(ValueError):
        merge_clusters([{0: 1.0}])
    with pytest.raises(ValueError):
        merge_clusters([{1: 2.0}, {}])


def test_two_blobs_end_to_end() -> None:
    rng = np.random.default_rng(1)
    x = np.zeros((40, 40), dtype=int)
    x[:20, :20] = rng.random((20, 20)) < 0.7
    x[20:, 20:] = rng.random((20, 20)) < 0.7
    labels = merge_clusters(candidate_scores(jaccard_distances(x)))
    assert len(set(labels[:20])) >= 1 and set(labels[:20]).isdisjoint(labels[20:])


def test_labels_are_contiguous_and_ordered_by_first_member() -> None:
    labels = merge_clusters([{}, {2: 1.0}, {1: 1.0}, {}])
    assert labels.tolist() == [0, 1, 1, 2]
