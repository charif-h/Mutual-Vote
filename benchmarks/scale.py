"""Time and memory benchmark for Mutual Vote on synthetic sparse binary user-item data.

Usage (one configuration per process, so memory numbers are not polluted):

    python benchmarks/scale.py N [--memory] [--ari]

The run is not instrumented, so the time is real. ``--memory`` also reports the process's
peak resident memory (Windows peak working set, or ``ru_maxrss`` elsewhere), which includes the
interpreter and the generated data (~100-200 MB). ``--ari`` reports the adjusted Rand index
against the generating groups (requires scikit-learn).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy import sparse

from mutualvote import cluster


def make_data(n: int, items: int = 2000, groups: int = 20, seed: int = 0):
    """Each user buys 12 items from its group's block of 80 items plus 4 random items."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, groups, n)
    rows, cols = [], []
    for u in range(n):
        base = (group[u] * 80) % (items - 80)
        chosen = np.concatenate(
            [base + rng.choice(80, 12, replace=False), rng.choice(items, 4, replace=False)]
        )
        rows += [u] * len(chosen)
        cols += chosen.tolist()
    data = np.ones(len(rows), dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, items)), group


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 1e6
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e6 if sys.platform == "darwin" else 1e3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--ari", action="store_true")
    args = parser.parse_args()

    x, truth = make_data(args.n)
    start = time.perf_counter()
    labels = cluster(x)
    elapsed = time.perf_counter() - start
    parts = [f"n={args.n}", f"clusters={labels.max() + 1}"]
    parts.append(f"time={elapsed:.1f}s")
    if args.memory:
        parts.append(f"peak_rss={peak_memory_mb():.0f} MB")
    if args.ari:
        from sklearn.metrics import adjusted_rand_score

        parts.append(f"ARI={adjusted_rand_score(truth, labels):.3f}")
    print(" ".join(parts), flush=True)


if __name__ == "__main__":
    main()
