# Decision record

Choices made where the paper is ambiguous or inconsistent. Each can be revisited; tests
should cite the number below.

## 1. Worked-example typo
The coupling list shows `score([B],[C]) = 0.5`, but the merge arithmetic uses `0.9`.
Only `0.9` reproduces the paper's printed results (0.425 and Ex = 0.7125).
**Decision:** use the arithmetic's numbers for the test vector; treat the list entry as a typo.

## 2. Neighbor threshold
The text removes users "larger than mean minus one standard deviation"; the pseudocode
keeps members "whose z-score is smaller than Z". Read literally these differ in sign
and direction.
**Decision:** keep `d <= mean - std` (z-score <= -1), the stated intent of keeping only
significantly close users. Equation (1) then uses `theta = mean - std`.

## 3. Mutual acceptance rule
The prose describes requests accepted only when the sender is at the top of the
receiver's list. The coupling-list formulation instead evaluates every linked pair and
merges by the average opinion.
**Decision:** the coupling list is the operative algorithm; the prose is intuition.

## 4. Member-pair accounting for merged opinions
The paper shows the merged opinion in an example but not the general rule.
**Decision:** the opinion of `X` about `Y` accumulates one interaction per member pair
`(x, y)`: positive with `c_x(y)` when `y` is a candidate of `x`, otherwise hostile with
`lambda`; prior `w = 2` is counted once. This reproduces the example. A cluster pair is
linked when at least one member pair has a candidate entry in either direction.
To keep merging cheap, store `(r, s, w-2)` sums per direction and add them on merge
instead of iterating member pairs.

## 5. One-sided pairs
If `B` is a candidate of `A` but `A` is not a candidate of `B`, `B`'s opinion of `A` is
a single negative interaction (eq. 7). Default `lambda = 0.5`, `a = 0.5`.

## 6. Degenerate cases (implementation choices, not in the paper)
* `theta == d_min` (all neighbor distances equal): give every neighbor score 1.
* No neighbors for a vector: it makes no invitations; it may still be chosen by others.
* Duplicate points (distance 0): treated as ordinary neighbors.
* Empty vectors under Jaccard: distance 0 between two empty vectors is ambiguous;
  define as 1 (maximally distant) so they never cluster by absence of data. Document in
  the distance module.
* `n < 2`: every point is its own cluster.

## 7. Small-cluster post-processing
The paper drops clusters with fewer than 5 users in its K-means experiments only.
**Decision:** not part of Mutual Vote; provide as an optional utility in the evaluation
module (issue #17).

## 8. Initial list values are candidate scores
In the worked example the initial list entries (1.0, 0.9, 0.8, ...) are used as `r`
increments, i.e. they are candidate scores `c`, while entries after the merge (0.425,
0.325, ...) are shifted opinion scores. Taken as opinion scores, a lone positive interaction
could never reach 1.0 (it gives `c/3`).
**Decision:** the algorithm orders pairs by the average *shifted opinion score*, also for
the initial singleton pairs. For `c1, c2` both positive that is `(c1 + c2) / 6`, which ranks
pairs exactly as the paper's example list does.

## 9. Standard deviation
The paper does not say population or sample. **Decision:** population std (`ddof=0`).
A small absolute tolerance (1e-12) is applied in `d <= theta` so equal distances are not
split by floating-point error.

## 10. Empty rows (Jaccard)
A row with no data is at distance 1 from everything, so its distance list has zero spread:
every other vector becomes a full-score candidate and it would merge into clusters.
**Decision:** `cluster(..., metric="jaccard")` sets empty rows aside as singleton clusters and
clusters the remaining rows (distances are computed without the empty rows).

## 11. Ties and row order
The closest neighbor of every vector scores exactly 1, so disjoint mutually-closest pairs
have exactly equal mean scores. Merging is greedy, so the order of tied merges can change
the result. **Decision:** ties are broken by cluster id (creation order), which makes runs
deterministic but means the output can depend on row order when exact ties occur. This is
inherent to the algorithm as described; row-permutation invariance is only tested on
tie-free candidate lists.

## 12. Cluster count is expected to be large
On the paper's dataset (~2400 users) Mutual Vote yields 53 clusters, about twice K-means'
24, with 100% coverage (Table 6, Section 4.3). On synthetic data with 5 groups of differing
density (750 vectors) this implementation gives 26-31 clusters and recovers dense groups
cleanly while sparser groups split. Treating a missing link between two clusters as one
hostile interaction (instead of one per member pair) gave 20-23 clusters, so the member-pair
rule (#4) is not the main cause. **Decision:** keep #4; fragmentation is consistent with
the paper's reported behaviour.
