# Edge co-variation representation protocol v1

## Question

Do groups of connectivity edges that co-vary across training participants provide a useful representation for classifying participants at an unseen acquisition group? The grouping is learned without diagnosis labels. This is a representation experiment, not a test of whether a discovered module is a biological disease mechanism.

## Representation

Each participant starts with 19,900 Fisher-z connectivity edges from the frozen CC200 cohort. Within every model-training partition, the transformer selects the 256 edges with the largest variance. It standardizes those edges using that partition, computes their across-participant Pearson correlation matrix, and groups edges using average-linkage clustering on distance `1 - abs(correlation)`. It sign-aligns each edge to the medoid of its module and summarizes a participant with the mean signed standardized edge value for each module.

The analysis requests 4, 8, or 16 modules. Every statistic used to select, standardize, correlate, or group edges is learned from the current training rows only. Labels are never passed to the transformer. The full transformation is nested inside the existing three-fold grouped model selection, so inner validation and outer held-out sites do not influence the learned modules.

The absolute correlation distance groups edges that co-vary positively or negatively. Sign alignment keeps inversely related edges from cancelling in the module summary. Module assignments can differ across training folds; they are not assumed to be stable anatomical systems. The report records assignment hashes, group sizes, and mean within-module absolute correlation for each fitted outer model.

## Comparisons

The frozen 432-participant cohort and nested leave-one-acquisition-group-out splits are reused. Raw and nuisance-residualized edges are analyzed separately. For each module count, the grouped representation is evaluated with tuned logistic regression and RBF SVM. A fidelity quantum kernel is evaluated on four modules only, with the same folds and labels. Existing PCA4 models and full-edge classical references provide context; hyperparameter selection uses only the inner grouped folds.

The primary metric is mean held-out-group balanced accuracy. AUROC is secondary. Paired comparisons between the four-module quantum kernel and its classical readouts resample the seven site-level differences; these intervals are descriptive because there are only seven groups and overlapping training sets. This is exploratory model selection over multiple module counts and comparisons, with no multiplicity correction.

## Interpretation limits

Across-participant correlations can reflect diagnosis mixture, site, demographics, motion, and other cohort structure. The raw/residual comparison is a sensitivity analysis and does not establish causality. Selecting high-variance edges and then clustering them is an explicit, compact module-learning procedure, but it may be unstable across sites. Any useful result requires examining fold-to-fold module stability and an independent cohort with compatible quality fields. The current ABIDE II metadata inventory does not support that external test.
