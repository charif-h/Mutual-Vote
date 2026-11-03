"""Command-line interface: ``mutualvote cluster INPUT [-o OUTPUT]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from . import __version__
from .estimator import MutualVote


def load_matrix(path: Path, delimiter: str | None = None) -> Any:
    """Load a matrix from ``.csv``/``.tsv``/``.txt`` (dense text), ``.npy`` or ``.npz``.

    ``.npz`` files are read as scipy sparse matrices (``scipy.sparse.save_npz``).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        return sparse.load_npz(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        if delimiter is None:
            delimiter = "\t" if suffix == ".tsv" else ","
        data = np.loadtxt(path, delimiter=delimiter, ndmin=2)
        return data
    raise ValueError(f"unsupported input format {suffix!r} (use .csv, .tsv, .txt, .npy or .npz)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mutualvote", description="Mutual Vote density clustering (Haydar & Boyer, UMAP'17)."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cluster", help="cluster the rows of a matrix and write one label per row")
    c.add_argument("input", type=Path, help=".csv/.tsv/.txt (dense), .npy, or .npz (scipy sparse)")
    c.add_argument("-o", "--output", type=Path, help="write labels here (default: stdout)")
    c.add_argument(
        "--metric", default="jaccard", help="jaccard (default), precomputed, or a scipy metric"
    )
    c.add_argument("--hostility", type=float, default=0.5, help="hostility barometer in [0, 1]")
    c.add_argument("--base-rate", type=float, default=0.5, help="subjective-logic base rate")
    c.add_argument("--chunk-size", type=int, default=1024, help="distance rows computed at a time")
    c.add_argument("--delimiter", help="text-file delimiter (default: ',' or tab for .tsv)")
    c.add_argument(
        "-v", "--verbose", action="store_true", help="report progress on stderr (tqdm if installed)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        x = load_matrix(args.input, args.delimiter)
        model = MutualVote(
            hostility=args.hostility,
            base_rate=args.base_rate,
            metric=args.metric,
            chunk_size=args.chunk_size,
            verbose=args.verbose,
        ).fit(x)
    except (OSError, ValueError) as exc:
        print(f"mutualvote: error: {exc}", file=sys.stderr)
        return 2
    text = "\n".join(str(int(label)) for label in model.labels_)
    if args.output is not None:
        args.output.write_text(text + ("\n" if text else ""), encoding="utf-8")
    elif text:
        print(text)
    print(f"{len(model.labels_)} rows, {model.n_clusters_} clusters", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
