"""The worked example of paper Section 3.2, checked against the slow reference."""

import pytest

from mutualvote.core import merge_clusters
from reference import coupling_list, reference_clusters

A, B, C, D = range(4)
# Candidate scores. score([B],[C]) is 0.9, the value the paper's arithmetic uses
# (docs/decisions.md #1); the first two columns of the paper's list are candidate scores (#8).
CANDS = [
    {B: 1.0, C: 0.8},  # A
    {A: 1.0, D: 0.9, C: 0.9},  # B
    {A: 0.6, B: 0.7},  # C
    {B: 0.7},  # D
]


def test_initial_top_pair_is_a_b() -> None:
    entries = coupling_list(CANDS, {i: (i,) for i in range(4)})
    assert (entries[0][1], entries[0][2]) == (A, B)
    # A-D is not linked. The order of the other pairs is not asserted: with score([B],[C])
    # = 0.9 (decisions.md #1) B-C outranks A-C, unlike the paper's printed list (0.5).
    assert {(e[1], e[2]) for e in entries} == {(A, B), (B, D), (A, C), (B, C)}


def test_scores_after_merging_a_and_b() -> None:
    clusters = {0: (A, B), 2: (C,), 3: (D,)}
    entries = {(e[1], e[2]): e for e in coupling_list(CANDS, clusters)}
    avg, _, _, ab_to_c, c_to_ab = entries[(0, 2)]
    assert ab_to_c == pytest.approx(0.425)
    assert c_to_ab == pytest.approx(0.325)
    assert avg == pytest.approx(0.375)
    avg, _, _, ab_to_d, d_to_ab = entries[(0, 3)]
    assert ab_to_d == pytest.approx(0.100)
    assert d_to_ab == pytest.approx(0.050)
    assert avg == pytest.approx(0.075)


def test_merge_history_and_final_partition() -> None:
    labels, history = reference_clusters(CANDS)
    assert [h[0] for h in history] == [(A, B), (A, B, C)]
    assert history[1][1] == pytest.approx(0.375)
    assert labels == [0, 0, 0, 1]  # D stays alone: both opinions about it turn negative


def test_fast_implementation_agrees() -> None:
    assert merge_clusters(CANDS).tolist() == [0, 0, 0, 1]
