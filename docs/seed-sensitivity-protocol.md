# Seed sensitivity

Specified on 2026-10-22 at 00:08 UTC, after the seed-42 low-data runs finished and while the full PCA, graph, and variational experiments were still running. Partial progress from those runs had been inspected. These are additional exploratory repetitions, not independent confirmations.

Use the fixed seeds 42, 137, and 2026. Retain all three and report each score as well as the mean and range across seeds. Do not select a seed from outer performance.

For the four-coordinate PCA comparisons, run every raw/residual and logistic/RBF/polynomial/quantum combination at training fractions 0.25, 0.5, and 1.0. Seed 42 is already covered by v2. The additional configurations are `config/sensitivity/pca-seed137.json` and `pca-seed2026.json`. They differ from v2 only in the seed and the full-data dimensionality list, which is restricted to four for these repetitions. The seed controls both randomized PCA and the nested stratified training subsets. It also controls solver initialization where relevant, so this measures their combined variability. The outer test participants and acquisition-group split rule stay fixed; inner grouped folds are recomputed from each sampled training set.

For the graph experiment, repeat the complete v4 raw/residual and four-readout comparison with encoder seeds 137 and 2026. Use `config/sensitivity/graph-seed137.json` and `graph-seed2026.json`; every other encoder setting stays unchanged. Kernel/readout settings keep the seed-42 v2 configuration so this comparison isolates encoder initialization, dropout, and minibatch order. Each readout still receives identical embeddings within its encoder seed and fold.

The additional outputs are `results/sensitivity-pca137-{quarter,half,full}/`, `sensitivity-pca2026-{quarter,half,full}/`, `sensitivity-graph137/`, and `sensitivity-graph2026/`. Preserve the original result series.

For an overall descriptive paired interval, first average each model's score across the three seeds within each held-out site. Then compare quantum and classical scores using the same site resamples. Do not treat the 21 seed/site cells as independent samples. Seven shared acquisition groups and overlapping outer training sets remain the uncertainty limit.
