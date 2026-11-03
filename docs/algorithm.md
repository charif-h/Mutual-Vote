# Mutual Vote: algorithm specification

Source: C. Haydar and A. Boyer, *A New Statistical Density Clustering Algorithm based on
Mutual Vote and Subjective Logic Applied to Recommender Systems*, UMAP'17,
doi:10.1145/3079628.3079676 (Section 3). Open points are tracked in
[decisions.md](decisions.md).

## Overview

Every vector starts as its own cluster. Each vector looks at its own distance
distribution to pick *candidates* it would like in its cluster, with a willingness
score. Candidate relations are turned into **subjective-logic opinions** between
clusters. The cluster pair with the highest mutual opinion is merged, opinions are
re-aggregated, and the process repeats until no pair has a positive mutual opinion.
There is no global distance threshold and no `k`.

## Step 1: distances

Pairwise distance between all vectors. The paper uses Jaccard distance (binary
purchase data); any metric works.

## Step 2: neighbors (per vector)

For each vector `u`, let `D(u)` be its distances to all other vectors.

```
theta_u = mean(D(u)) - std(D(u))
neighbors(u) = { v : d(u, v) <= theta_u }
```

Equivalent to keeping z-score <= -1. The threshold is local to each vector.

## Step 3: candidates, equation (1)

For each neighbor `n` of `u`, with `d_min` the distance to `u`'s closest vector:

```
c_u(n) = 1 - (d(u, n) - d_min) / (theta_u - d_min)
```

Scores lie in [0, 1]; the closest neighbor scores 1. `candidates(u)` maps each
neighbor to its score. A vector `v` not in `candidates(u)` is *not wanted* by `u`.

## Step 4: binomial opinions (subjective logic)

An opinion has counters `r` (positive), `s` (negative), weight `w` and base rate `a`.
Initial state: `r = s = 0`, `w = 2`, `a = 0.5`.

```
b = r / (r + s + w)        belief          (2)
d = s / (r + s + w)        disbelief       (3)
u = w / (r + s + w)        uncertainty     (4)
Ex = b + a * u             expected value  (5)
score = 2 * Ex - 1         shifted to [-1, +1]
```

Updates:

| Interaction | Update |
|---|---|
| Positive, with candidate score `c` (eq. 6) | `r += c`, `w += 1 - c` |
| Negative / hostile, with barometer `lambda` (eq. 7) | `s += lambda`, `w += 1 - lambda` |

Defaults: `a = 0.5`, `lambda = 0.5`.

Check (Table 2): initial -> (b,d,u) = (0,0,1); one full positive (`c = 1`) -> (1/3, 0, 2/3);
then one full negative (`lambda = 1`) -> (1/4, 1/4, 2/4).

## Step 5: coupling list

The opinion of cluster `X` about cluster `Y` starts from the prior (`r=s=0, w=2`) and
accumulates one interaction per member pair `(x, y)` with `x in X`, `y in Y`:

* `y in candidates(x)` -> positive interaction with `c_x(y)`;
* otherwise -> negative interaction with `lambda`.

A cluster pair is *linked* if at least one member pair has a candidate entry in either
direction. Each linked pair holds `(opinion X->Y, opinion Y->X)` and their average
`avg = (score(X->Y) + score(Y->X)) / 2`. The list is ordered by `avg` descending.

Initially every cluster is a single vector, so each opinion has one interaction. A single
positive interaction with candidate score `c` gives shifted score `c / 3`; a single hostile
one (`lambda = 0.5`) gives `-1/6`. Ordering and the stop test always use these shifted
opinion scores (see decisions.md #8).

## Step 6: merging loop (Algorithm 1)

```
repeat
    take the top pair (X, Y) of the coupling list
    merge X and Y into one cluster
    recompute opinions between the merged cluster and every cluster linked to X or Y
    remove the old entries, reorder by avg
until the best avg < 0   (or the list is empty)
```

Vectors left in singleton clusters are valid clusters (no forced assignment). Links
whose `avg` is negative are dropped from the list.

## Worked example (paper, Section 3.2): test vectors

Initial list. The first two columns are the **candidate scores** `c` (not opinion scores;
the paper's list mixes the two, see decisions.md #8), the last is their average:

```
{A,B}: 1.0, 1.0, 1.0
{B,D}: 0.9, 0.7, 0.8
{A,C}: 0.8, 0.6, 0.7
{B,C}: 0.5, 0.7, 0.6     <- see decisions.md #1: merge arithmetic uses 0.9 for score([B],[C])
```

Merge `[A]` and `[B]`:

* `score([A,B],[C])`: `r = 0.8 + 0.9 = 1.7`, `s = 0`, `w = 2 + 0.2 + 0.1 = 2.3`;
  `b = 0.425`, `d = 0`, `u = 0.575`; `Ex = 0.7125`; `score = 0.425`.
* `score([A,B],[D])`: `r = 0.9`, `s = 0.5` (A has no opinion of D),
  `w = 2 + 0.1 + 0.5 = 2.6`; total `4.0`; `b = 0.225`, `u = 0.65`; `Ex = 0.55`; `score = 0.1`.

After the merge, the entries are true shifted opinion scores (`A->B`, `B->A`, avg): `({[A,B],[C]}: 0.425, 0.325, 0.375)` and
`({[A,B],[D]}: 0.100, 0.050, 0.075)`.

## Complexity note

Naively the list holds up to `O(n * |neighbors|)` linked pairs. The implementation
should use a heap with lazy deletion and per-cluster adjacency (issue #10) rather than
re-sorting the list each iteration.
