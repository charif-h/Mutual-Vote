from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mutualvote import cluster
from mutualvote.cli import load_matrix, main


def data() -> np.ndarray:
    rng = np.random.default_rng(1)
    x = np.zeros((30, 30), dtype=int)
    x[:15, :15] = rng.random((15, 15)) < 0.7
    x[15:, 15:] = rng.random((15, 15)) < 0.7
    return x


def test_csv_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    x = data()
    path = tmp_path / "m.csv"
    np.savetxt(path, x, fmt="%d", delimiter=",")
    assert main(["cluster", str(path)]) == 0
    out, err = capsys.readouterr()
    assert [int(v) for v in out.split()] == cluster(x).tolist()
    assert "30 rows" in err


def test_npz_sparse_to_file(tmp_path: Path) -> None:
    x = data()
    src, dst = tmp_path / "m.npz", tmp_path / "labels.txt"
    sparse.save_npz(src, sparse.csr_matrix(x))
    assert main(["cluster", str(src), "-o", str(dst)]) == 0
    assert [int(v) for v in dst.read_text().split()] == cluster(x).tolist()


def test_npy_and_tsv_and_options(tmp_path: Path) -> None:
    x = data()
    np.save(tmp_path / "m.npy", x)
    np.savetxt(tmp_path / "m.tsv", x, fmt="%d", delimiter="\t")
    expected = cluster(x, hostility=1.0, chunk_size=7).tolist()
    for name in ("m.npy", "m.tsv"):
        dst = tmp_path / f"{name}.out"
        args = ["cluster", str(tmp_path / name), "-o", str(dst), "--hostility", "1.0"]
        assert main([*args, "--chunk-size", "7"]) == 0
        assert [int(v) for v in dst.read_text().split()] == expected


def test_precomputed_metric(tmp_path: Path) -> None:
    pts = np.random.default_rng(0).random((20, 2))
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    np.savetxt(tmp_path / "d.csv", d, delimiter=",")
    dst = tmp_path / "out.txt"
    assert (
        main(["cluster", str(tmp_path / "d.csv"), "-o", str(dst), "--metric", "precomputed"]) == 0
    )
    expected = cluster(d, metric="precomputed").tolist()
    assert [int(v) for v in dst.read_text().split()] == expected


def test_errors_return_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["cluster", str(tmp_path / "missing.csv")]) == 2
    (tmp_path / "x.parquet").write_text("x")
    assert main(["cluster", str(tmp_path / "x.parquet")]) == 2
    np.savetxt(tmp_path / "m.csv", data(), fmt="%d", delimiter=",")
    assert main(["cluster", str(tmp_path / "m.csv"), "--hostility", "3"]) == 2
    assert "error" in capsys.readouterr().err


def test_load_matrix_single_row(tmp_path: Path) -> None:
    (tmp_path / "r.csv").write_text("1,0,1\n")
    assert load_matrix(tmp_path / "r.csv").shape == (1, 3)


def test_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
