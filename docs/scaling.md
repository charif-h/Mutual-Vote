# Scaling

How time and memory grow with the number of vectors, measured with
[`benchmarks/scale.py`](../benchmarks/scale.py).

## Measurements

Synthetic sparse binary user-item data: 2000 items, 20 groups, each user buys 12 items from
its group's block of 80 items plus 4 random items. Python 3.11, one process, Windows 11, Intel i7-12700H, 32 GB RAM.
`peak RSS` is the process's peak resident memory and includes the interpreter and the data
(about 100-150 MB).

| vectors | time | peak RSS | clusters | ARI vs groups |
|---:|---:|---:|---:|---:|
| 1,000 | 0.4 s | 139 MB | 20 | 1.000 |
| 2,400 (the paper's dataset size) | 2.6 s | 307 MB | 20 | 1.000 |
| 5,000 | 14.3 s | 898 MB | 20 | 1.000 |
| 10,000 | 88 s | 3.3 GB | 20 | 1.000 |

Reproduce: `python benchmarks/scale.py 10000 --memory --ari`.

These groups are cleanly separated, so ARI is 1.0 at every size. The numbers show cost, not
clustering quality on hard data (see [decisions.md](decisions.md) #12 for fragmentation).
Timings vary by machine and by load: a few runs made while the machine was busy took about
2x longer.

## Where the cost is

The number of *candidate pairs* grows roughly with n^2: every vector keeps all neighbors below
its own mean minus one standard deviation, and on this data that is a constant fraction of
the dataset. The merge loop's time and memory follow that count (about 230 candidates per
vector at n = 2,400). Memory grows about 4x per doubling of n, time about 5-6x.

* **Distances** are cheap and use bounded memory: they are computed `chunk_size` rows at a
  time (default 1024) and only each vector's candidate list is kept, so the full n x n matrix
  is never built. `chunk_size` changes memory only, never the result.
* **The merge loop** dominates. It uses a heap with lazy deletion and per-cluster adjacency,
  with opinions stored as plain tuples; this was about 3x faster than an earlier version
  built on `Opinion` objects (6.7 s to 2.4 s at n = 2,400).

## Practical limits

* Up to about 5,000 vectors: seconds to under half a minute and under 1 GB.
* 10,000 vectors: about a minute and a half and 3.3 GB.
* Beyond that, extrapolating the n^2 growth, 20,000 vectors would need roughly 13 GB and
  several minutes. This is an extrapolation, not a measurement. Real data with fewer
  neighbors per vector will use less.

## Watching a run: `verbose=True`

Verbose mode reports the stage and, while candidates are computed, the pairs found so far and
a projected memory for the merge stage (`BYTES_PER_PAIR` = 360, measured at n = 2,400: 553,690
pairs, 202 MB above baseline, about 364 bytes per pair). The projection extrapolates the pairs
found so far to all rows, so it is an order-of-magnitude warning that shows up after the first
few percent of rows, not a guarantee. If it is far above your free RAM, subsample the data
or stop the run: the machine will start swapping and the merge stage will not finish in
reasonable time.

Large datasets such as MNIST (60,000-70,000 vectors) are far beyond the sizes measured
above (the n^2 growth puts them in the hundreds of GB), so start from a subsample of 5,000 to
10,000 rows.

## What was tried and rejected

Capping the number of candidates per vector looked like an easy fix. On 10,000 vectors it
cut the time to 5-12 s but destroyed the clustering: keeping the 200 closest candidates gave
628 clusters (ARI 0.066) and 50 gave 1,956 (ARI 0.018), against 20 clusters and ARI 1.000
uncapped. Truncating makes most candidate links one-sided, and a missing opinion counts as
hostile. The option was removed; see [decisions.md](decisions.md) #13.
