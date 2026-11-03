# mutualvote

Python implementation of the **Mutual Vote** clustering algorithm: a density-based
clustering method that adapts to each vector's local neighborhood and lets vectors
"vote" mutually, using subjective logic, on whether to share a cluster. It needs
almost no parameters and handles clusters of different densities, which makes it
suited to sparse user-item data in recommender systems.

> Status: under construction (see the milestones and issues on GitHub).

Based on: C. Haydar and A. Boyer, *A New Statistical Density Clustering Algorithm
based on Mutual Vote and Subjective Logic Applied to Recommender Systems*, UMAP'17,
doi:[10.1145/3079628.3079676](https://doi.org/10.1145/3079628.3079676).

See [docs/algorithm.md](docs/algorithm.md) for the specification this library implements.

## Usage

```python
import numpy as np
from mutualvote import MutualVote

X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]])  # users x items
labels = MutualVote().fit_predict(X)  # -> [0, 0, 1, 1]
```

`MutualVote` follows the scikit-learn estimator API (`fit`, `fit_predict`, `labels_`,
`n_clusters_`, `get_params`/`set_params`) and works with scikit-learn tools such as `clone` and
`Pipeline` when scikit-learn is installed; it does not require it. Input can be a dense array,
a scipy sparse matrix, or a precomputed distance matrix (`metric="precomputed"`).

Mutual Vote typically produces many fine-grained clusters (the paper reports about twice as
many as K-means); see [docs/decisions.md](docs/decisions.md) #12. Rows with no data become
singleton clusters.

From the command line:

```bash
mutualvote cluster purchases.csv -o labels.txt     # also .tsv, .npy, or sparse .npz
```

For time and memory at 1k-10k vectors see [docs/scaling.md](docs/scaling.md).

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```
