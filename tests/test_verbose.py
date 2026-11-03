from pathlib import Path

import numpy as np
import pytest

from mutualvote import MutualVote, _progress, cluster
from mutualvote.candidates import BYTES_PER_PAIR, _pairs_info
from mutualvote.cli import main


def data() -> np.ndarray:
    rng = np.random.default_rng(1)
    x = np.zeros((40, 30), dtype=int)
    x[:20, :15] = rng.random((20, 15)) < 0.7
    x[20:, 15:] = rng.random((20, 15)) < 0.7
    return x


@pytest.fixture
def text_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the dependency-free fallback and report on every update."""
    monkeypatch.setattr(_progress, "_tqdm", None)
    monkeypatch.setattr(_progress, "FALLBACK_INTERVAL", 0.0)


def test_silent_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    cluster(data())
    MutualVote().fit(data())
    out, err = capsys.readouterr()
    assert out == "" and err == ""


@pytest.mark.parametrize("metric", ["jaccard", "euclidean"])
def test_verbose_does_not_change_labels(
    metric: str, text_progress: None, capsys: pytest.CaptureFixture[str]
) -> None:
    x = data()
    quiet = MutualVote(metric=metric, chunk_size=7).fit_predict(x)
    loud = MutualVote(metric=metric, chunk_size=7, verbose=True).fit_predict(x)
    assert quiet.tolist() == loud.tolist()
    assert capsys.readouterr().err != ""


def test_text_report_covers_every_stage(
    text_progress: None, capsys: pytest.CaptureFixture[str]
) -> None:
    cluster(data(), chunk_size=16, verbose=True)
    err = capsys.readouterr().err
    assert "40 vectors, 30 features" in err
    assert "distances and candidates (rows): 40/40 (100.0%)" in err
    assert "candidate pairs for 40 vectors" in err
    assert "merge stage projected" in err
    assert "linking (rows)" in err
    assert "merging (clusters merged away)" in err
    k = cluster(data(), chunk_size=16).max() + 1
    assert f"done: {k} clusters after {40 - k} merges" in err
    assert all(line.startswith("[mutualvote] ") for line in err.splitlines())


def test_empty_rows_are_reported(text_progress: None, capsys: pytest.CaptureFixture[str]) -> None:
    x = data()
    x[3] = 0
    cluster(x, verbose=True)
    assert "1 empty rows set aside" in capsys.readouterr().err


def test_tqdm_is_used_when_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tqdm_auto = pytest.importorskip("tqdm.auto")
    monkeypatch.setattr(_progress, "_tqdm", tqdm_auto.tqdm)
    cluster(data(), verbose=True)
    err = capsys.readouterr().err
    assert "mutualvote merging" in err
    assert "mutualvote linking" in err


def test_fallback_is_throttled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_progress, "_tqdm", None)
    monkeypatch.setattr(_progress, "FALLBACK_INTERVAL", 3600.0)
    cluster(data(), chunk_size=4, verbose=True)
    err = capsys.readouterr().err
    # no periodic lines, only the closing line of each bar: 10 blocks would otherwise show up
    assert err.count("distances and candidates (rows):") == 1


def test_pairs_info_projects_memory() -> None:
    # 1,000 pairs after 10 of 100 rows -> 10,000 pairs at the end
    text = _pairs_info(1000, 10, 100)
    assert "1,000 pairs" in text
    assert f"{10_000 * BYTES_PER_PAIR / 1e9:,.1f} GB" in text
    assert _pairs_info(5, 5, None) == "5 pairs"


def test_format_duration() -> None:
    assert _progress.format_duration(75) == "01:15"
    assert _progress.format_duration(3725) == "1:02:05"


def test_cli_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    np.savetxt(tmp_path / "m.csv", data(), fmt="%d", delimiter=",")
    assert main(["cluster", str(tmp_path / "m.csv"), "-o", str(tmp_path / "o.txt"), "-v"]) == 0
    err = capsys.readouterr().err
    assert "linking" in err and "done:" in err
