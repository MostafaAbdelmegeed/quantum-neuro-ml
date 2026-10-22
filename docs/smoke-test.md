# Connectivity smoke test

This check exercises metadata parsing, derivative downloads, connectivity construction, and a site holdout on a small cohort. It fits no model and produces no performance estimate.

## Run

From the repository root:

```bash
uv sync --locked --extra dev --python 3.11
uv run --locked --extra dev python -m pytest
uv run --locked --extra dev qneuro-abide-smoke
uv run --locked --extra dev qneuro-abide-smoke --offline
```

The first run downloads metadata if the metadata file is absent. Later runs reuse that snapshot. `--metadata` can select another local snapshot. Offline mode requires the metadata and all selected derivatives to exist locally with valid checksum receipts; it makes no network requests.

## Cohort selection

The default cohort contains eight participants: the two lowest available subject IDs in each diagnosis group at NYU and PITT. Selection is deterministic and independent of source row order. PITT supplies the four test participants; NYU supplies the four training participants. Duplicate participant IDs or file IDs cause an error.

`--sites`, `--held-out-site`, and `--per-group` change this integration cohort. At least two distinct sites must be present, and every selected site must contain enough participants in both diagnosis groups. A failed download or invalid signal stops the run rather than substituting another participant.

These choices do not define the research cohort. [Cohort policy v1](cohort-policy.md) separately fixes motion exclusions, imaging quality ratings, required covariates, and minimum acquisition-group size. The [PCP quality assessment documentation](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/quality_assessment.md) describes the available motion metrics and manual ratings.

## Connectivity contract

The input configuration is C-PAC / `filt_noglobal` / `rois_cc200`, following the [PCP download layout](https://preprocessed-connectomes-project.org/abide/download.html).

Each file must have ordered ROI labels `#1` through `#200`, 200 numeric columns, and at least three timepoints. Three timepoints is a numerical guard, not a scientific scan-duration threshold. Non-finite values and constant ROI signals cause an error. The pipeline does not impute missing values or drop ROI columns.

For each participant, the pipeline computes Pearson correlation between ROI columns using [NumPy's `corrcoef`](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html). It makes the result symmetric and sets the correlation diagonal to one. Fisher-z conversion clips correlations to `[-1 + 1e-7, 1 - 1e-7]` before `arctanh` and sets the diagonal to zero. Negative edges remain negative. The upper triangle contains 19,900 features in NumPy row order.

Each participant's connectivity uses only that participant's scan. The pipeline performs no cohort scaling, feature selection, dimensionality reduction, graph threshold fitting, or nuisance regression. Later modeling code must fit those operations inside training folds.

## Local outputs

All paths below are ignored by Git:

| Path | Contents |
| --- | --- |
| `data/metadata/Phenotypic_V1_0b_preprocessed1.csv` | Cached metadata snapshot |
| `data/raw/cpac/filt_noglobal/rois_cc200/` | Downloaded `.1D` files and JSON checksum receipts |
| `data/smoke/cohort.csv` | Selected subject metadata and derivative URLs |
| `data/smoke/connectivity.npz` | Matrices, edge features, labels, identities, and split mask |
| `data/smoke/report.json` | Source and output digests, configuration, counts, and per-scan dimensions |

The NPZ archive contains `correlation` and `fisher_z` with shape `(subjects, 200, 200)`, `features` with shape `(subjects, 19900)`, and aligned arrays `subject_ids`, `site_ids`, `labels`, and `is_test`. `labels` uses autism = 1 and control = 0; `is_test` marks the held-out site. `roi_labels` fixes the ROI axis order. Load the archive with `numpy.load(..., allow_pickle=False)`.

Checksums record the bytes received and detect later local changes. They are not publisher-provided integrity guarantees. A cached file must match its saved URL and SHA-256 receipt before reuse. If a cache entry is incomplete or mismatched, inspect it and remove that file and its receipt before downloading again.

Only use outputs after the command exits successfully. A failed rerun can leave outputs from the previous successful run. Subject manifests and imaging-derived arrays stay local; reports committed to the repository should contain aggregate results only.
