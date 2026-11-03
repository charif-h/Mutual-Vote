import numpy as np
import pytest
from scipy import sparse

from mutualvote import MutualVote, cluster
from mutualvote._compat import HAS_SKLEARN, _FallbackBaseEstimator, _FallbackClusterMixin


def blobs() -> np.ndarray:
    rng = np.random.default_rng(1)
    x = np.zeros((40, 40), dtype=int)
    x[:20, :20] = rng.random((20, 20)) < 0.7
    x[20:, 20:] = rng.random((20, 20)) < 0.7
    return x


def test_fit_sets_attributes_and_returns_self() -> None:
    x = blobs()
    est = MutualVote()
    assert est.fit(x) is est
    assert est.labels_.shape == (40,)
    assert est.n_clusters_ == est.labels_.max() + 1
    assert est.n_features_in_ == 40


def test_fit_predict_equals_functional_api() -> None:
    x = blobs()
    assert MutualVote().fit_predict(x).tolist() == cluster(x).tolist()


def test_parameters_are_forwarded() -> None:
    x = blobs()
    hostile = MutualVote(hostility=1.0).fit_predict(x)
    assert hostile.tolist() == cluster(x, hostility=1.0).tolist()


def test_sparse_dense_and_precomputed_inputs() -> None:
    x = blobs()
    dense = MutualVote().fit_predict(x)
    assert MutualVote().fit_predict(sparse.csr_matrix(x)).tolist() == dense.tolist()
    pts = np.random.default_rng(0).random((20, 2))
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    est = MutualVote(metric="precomputed").fit(d)
    assert est.labels_.tolist() == MutualVote(metric="euclidean").fit_predict(pts).tolist()
    assert not hasattr(est, "n_features_in_")


def test_empty_input() -> None:
    est = MutualVote().fit(np.zeros((0, 3), dtype=int))
    assert est.n_clusters_ == 0 and est.labels_.size == 0


def test_get_and_set_params() -> None:
    est = MutualVote(hostility=0.2)
    assert est.get_params() == {
        "base_rate": 0.5,
        "chunk_size": 1024,
        "hostility": 0.2,
        "metric": "jaccard",
        "verbose": False,
    }
    assert est.set_params(hostility=0.7) is est
    assert est.hostility == 0.7
    with pytest.raises(ValueError):
        est.set_params(nope=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hostility": -0.1},
        {"hostility": 1.5},
        {"base_rate": 2},
        {"metric": 3},
        {"chunk_size": 0},
    ],
)
def test_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MutualVote(**kwargs).fit(blobs())  # type: ignore[arg-type]


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
def test_works_with_scikit_learn_tools() -> None:
    from sklearn.base import ClusterMixin, clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    est = MutualVote(hostility=0.3)
    assert isinstance(est, ClusterMixin)
    copy = clone(est)
    assert copy is not est and copy.get_params() == est.get_params()
    x = blobs()
    labels = make_pipeline(FunctionTransformer(), MutualVote()).fit_predict(x)
    assert labels.tolist() == MutualVote().fit_predict(x).tolist()
    assert set(labels[:20]).isdisjoint(labels[20:])  # no cluster mixes the two blobs


def test_fallback_base_classes_work_without_scikit_learn() -> None:
    class Toy(_FallbackClusterMixin, _FallbackBaseEstimator):
        def __init__(self, a: int = 1, b: str = "x") -> None:
            self.a = a
            self.b = b

        def fit(self, X: object, y: object = None) -> "Toy":  # noqa: N803
            self.labels_ = np.array([self.a])
            return self

    toy = Toy(a=3)
    assert toy.get_params() == {"a": 3, "b": "x"}
    assert toy.set_params(b="y").b == "y"
    assert toy.fit_predict(None).tolist() == [3]
    assert repr(toy) == "Toy(a=3, b='y')"
    with pytest.raises(ValueError):
        toy.set_params(zzz=1)
