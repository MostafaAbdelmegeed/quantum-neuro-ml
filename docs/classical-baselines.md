# Classical reference experiment

The first experiment compares L2 logistic regression, RBF SVM, and a training-prior dummy control on Fisher-z CC200 connectivity edges. The [cohort policy](cohort-policy.md) fixes the inclusion rules, search grids, and evaluation before training.

## Reproduce

Run from the repository root with Python 3.11:

```bash
uv sync --locked --extra dev --python 3.11
uv run --locked --extra dev python -m pytest
uv run --locked --extra dev qneuro-abide-cohort
uv run --locked --extra dev qneuro-abide-build --workers 4
uv run --locked --extra dev qneuro-abide-baseline
```

The cohort command needs the pinned local metadata snapshot at `data/metadata/Phenotypic_V1_0b_preprocessed1.csv`. A fresh checkout can create it without imaging downloads:

```bash
uv run --locked --extra dev python -c "from pathlib import Path; from quantum_neuro_ml.data.abide import fetch_phenotypic_csv; p = Path('data/metadata/Phenotypic_V1_0b_preprocessed1.csv'); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(fetch_phenotypic_csv().encode('utf-8'))"
```

If that metadata fingerprint differs from the frozen snapshot, the audit stops. Resolve the source change and create a new policy version rather than silently changing this experiment.

The dataset build downloads metadata-eligible scans, checks checksum receipts, validates the signals, and applies site minimums again. It writes `data/cohort/v1/connectivity.npz`, `manifest.csv`, `decisions.csv`, `audit.json`, and `report.json`. The archive contains aligned features, labels, participant IDs, original site labels, acquisition groups, age, sex, and mean FD.

To rebuild from cached scans, choose a separate destination:

```bash
uv run --locked --extra dev qneuro-abide-build --offline --output-dir data/cohort/v1-offline
```

A completed dataset build cannot overwrite an existing dataset. A failed download can be retried with the same command because valid cached files remain available. The modeling command also requires a new output directory for each completed result series; `--output-dir results/classical-v1-repeat` selects one. Use outputs only after a successful exit and the final `report.json` is present.

## Folds and metrics

The outer evaluation holds out an entire acquisition group. Three inner grouped folds select parameters using the remaining groups. Scikit-learn's [pipeline guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) explains why scaling must be inside the search, and [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) defines the split behaviour.

The runner refuses duplicate participants, fewer than four acquisition groups, or a group missing either diagnosis. It verifies the dataset digest and policy fingerprint and checks that repeated site samples have been grouped together. Convergence warnings stop a run.

`results/classical-v1/report.json` contains fold membership at group level, training and test counts, selected parameters, inner validation scores, held-out metrics, software versions, and elapsed time. `predictions.csv` contains one held-out prediction per participant and model. Both stay outside Git. The report records the metadata, manifest, dataset, and prediction digests.

The primary summaries give each acquisition group equal weight. The pooled metrics weight participants equally and are descriptive; pooled SVM AUROC also mixes score scales from separately fitted models. The 95% intervals bootstrap the seven observed groups with replacement and take percentile bounds over 2,000 resamples. Shared training data across outer folds and the small number of sites limit their interpretation.

SVM decision values support AUROC, but the first run does not calibrate SVM probabilities. Its Brier score and expected calibration error are `null`. Logistic regression and the dummy control report both. The dummy predicts the training majority class and uses the training class prior as its probability.

## Limits of this run

This is a connectivity reference experiment. It does not yet implement graph encoders, the quantum dimensional bottleneck, nuisance residualisation, repeated secondary evaluation, or quantum models. Age, sex, residual motion, and selection from strict QC can still affect the results. Treat the site-wise variation as part of the result, and do not use it to change the frozen inclusion policy or parameter grids.
