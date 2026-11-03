"""Cross-check the optimized merge loop against the slow reference on random inputs."""

import numpy as np
import pytest

from mutualvote.candidates import candidate_scores
from mutualvote.core import merge_clusters
from reference import reference_clusters


def random_candidates(rng: np.random.Generator, n: int, density: float) -> list[dict[int, float]]:
    # continuous random scores: no exact ties, so tie-breaking cannot make runs differ
    return [
        {v: float(rng.random()) for v in range(n) if v != u and rng.random() < density}
        for u in range(n)
    ]


@pytest.mark.parametrize("seed", range(150))
def test_random_candidate_lists(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 33))
    cands = random_candidates(rng, n, float(rng.uniform(0.1, 0.6)))
    hostility = float(rng.choice([0.0, 0.3, 0.5, 1.0]))
    base = float(rng.choice([0.25, 0.5]))
    expected, _ = reference_clusters(cands, hostility, base)
    assert merge_clusters(cands, hostility, base).tolist() == expected


@pytest.mark.parametrize("seed", range(40))
def test_random_distance_matrices(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(3, 25))
    points = rng.random((n, 2)) * rng.choice([1.0, 5.0], size=(1, 2))
    dist = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    cands = candidate_scores(dist)
    expected, _ = reference_clusters(cands)
    assert merge_clusters(cands).tolist() == expected
