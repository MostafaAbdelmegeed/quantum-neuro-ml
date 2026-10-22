# Classical reference results v1

Run date: 2026-10-22. Logistic regression and RBF SVM each reached about 69.1% balanced accuracy and 0.78 AUROC, averaged equally across seven held-out acquisition groups. Their mean scores are almost identical.

The experiment uses the 432-participant [validated cohort](cohort-audit.md): 205 autism and 227 control participants. The [cohort policy](cohort-policy.md) fixed inclusion rules and search grids before training. Each outer fold holds out an acquisition group, and three inner grouped folds select parameters. Scaling is inside the fitted pipeline throughout. The inputs are 19,900 Fisher-z connectivity edges per participant.

## Primary results

These are unweighted means across acquisition groups. Parentheses give the descriptive 95% site-bootstrap interval from 2,000 resamples.

| Model | Balanced accuracy | AUROC |
| --- | ---: | ---: |
| Training-prior dummy | 50.00% (50.00-50.00%) | 0.500 (0.500-0.500) |
| L2 logistic regression | 69.06% (64.59-73.32%) | 0.781 (0.749-0.810) |
| RBF SVM | 69.05% (63.65-73.39%) | 0.780 (0.753-0.803) |

The intervals summarise variation among seven observed groups. The outer training sets overlap, and this study has no independent validation cohort. These intervals do not establish a significant model difference.

| Held-out group | Participants | Logistic balanced accuracy | Logistic AUROC | SVM balanced accuracy | SVM AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LEUVEN | 32 | 60.00% | 0.713 | 55.83% | 0.725 |
| NYU | 145 | 73.56% | 0.792 | 73.56% | 0.811 |
| TRINITY | 36 | 77.71% | 0.836 | 72.76% | 0.799 |
| UCLA | 42 | 70.25% | 0.732 | 68.08% | 0.732 |
| UM | 101 | 62.43% | 0.803 | 66.46% | 0.796 |
| USM | 49 | 67.10% | 0.809 | 69.30% | 0.783 |
| YALE | 27 | 72.35% | 0.782 | 77.35% | 0.812 |

Performance varies across acquisition groups. For example, logistic regression's balanced accuracy ranges from 60.0% in Leuven to 77.7% in Trinity. Michigan's AUROC is 0.803 but its balanced accuracy is 62.4%, so a ranking score alone does not describe threshold behaviour.

## Other metrics and tuning

| Model | Mean macro F1 | Mean MCC | Mean Brier score | Mean 10-bin calibration error |
| --- | ---: | ---: | ---: | ---: |
| Training-prior dummy | 0.315 | 0.000 | 0.254 | 0.102 |
| Logistic regression | 0.672 | 0.407 | 0.212 | 0.193 |
| RBF SVM | 0.662 | 0.399 | Not estimated | Not estimated |

SVM AUROC uses decision scores; the run does not calibrate SVM probabilities. Logistic regression improves Brier score over the dummy, but its calibration error remains material and estimates from small held-out groups are noisy.

Logistic regression selects `C=0.001` for Leuven, Michigan, USM, and Yale; `C=0.01` for Trinity; and `C=0.1` for NYU and UCLA. SVM selects the largest tested `C=10` for six groups and `C=1` for Yale. Its selected gamma is `0.1/p` for Leuven, NYU, Trinity, UCLA, and Michigan, and `1/p` for USM and Yale. The frequent upper-bound C selection warrants a separately specified wider-grid experiment. The v1 grid and results remain fixed.

Pooled participant-level balanced accuracy is 69.01% for logistic regression and 68.58% for SVM. Pooled AUROC is 0.761 and 0.736 respectively. Those summaries weight participants equally and mix models fitted in different folds; pooled SVM scores also have fold-specific scales. Use the group means above as the primary comparison.

## Reproducibility and limits

The run took 193 seconds on the local Windows machine with one numerical-library thread. Versions were Python 3.11.15, NumPy 2.4.6, and scikit-learn 1.9.1. The local report records the processor, full grids, selected parameters, inner group membership, timings, source-file hashes, and lockfile fingerprint. [Reproduction commands](classical-baselines.md) describe the workflow.

An independent check recomputed balanced accuracy from confusion counts and AUROC from score ranks for every model/site pair. It also checked each participant's diagnosis and group against the dataset, confirmed one held-out prediction per model, and verified group separation, prediction digests, source hashes, and the lockfile. The automated suite passes all 38 tests, including metadata-to-model integration, scaler-fit isolation, and cache/dataset corruption checks. An offline dataset rebuild reproduced all arrays, manifests, and exclusion decisions exactly.

Participant arrays and predictions remain local under `data/` and `results/`. The prediction CSV contains 1,296 rows, one for each participant/model pair. Its SHA-256 is `c6e1508beac70d27312a109302f83da5445525a88689c0efe1e2b54231400ea3`. Policy and dataset fingerprints are recorded in the [cohort audit](cohort-audit.md), and dataset citations and terms are in [data use](data-use.md).

This reference cohort retains 41.7% of participants with derivative identifiers. Residual motion and sex composition differ by diagnosis, and the models do not adjust for those covariates. These results therefore do not isolate disease-specific connectivity effects. The next controls should measure demographic/motion-only prediction and fit nuisance adjustments within training folds. Graph encoders, the shared low-dimensional bottleneck, and quantum comparisons remain unimplemented.
