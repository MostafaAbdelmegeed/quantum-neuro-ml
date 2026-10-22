# Follow-up experiments v2

Specified on 2026-10-22 after observing the v1 reference results, before running the experiments below. These are exploratory follow-ups. Reusing the same seven sites does not create a new blinded validation cohort. The [cohort policy](cohort-policy.md) stays unchanged, and every run records the configuration and source before fitting.

The executable settings are [experiments-v2.json](../config/experiments-v2.json). Report every specified representation and dimension, including negative results. Do not select a favourable site, circuit depth, or training fraction from outer test scores.

## Nuisance controls

Compare age/age-squared/sex/motion-only logistic regression and RBF SVM with connectivity models. The four covariate features are age, age squared, an indicator for source sex code 2, and mean FD. Fit scaling inside each training fold. This tests whether these recorded variables alone predict diagnosis under the acquisition-group split.

For a separate sensitivity experiment, regress each connectivity edge on those four covariates and an intercept using all training participants. Estimate coefficients in each inner training partition and again in the outer training data before refitting. Apply the coefficients unchanged to validation/test participants. Scale the resulting residuals inside the pipeline. This may remove signal associated with both nuisance variables and diagnosis; it is not a causal adjustment or proof that retained signal is disease-specific. Do not include a site dummy or estimate harmonisation parameters from the held-out site.

Run full-edge logistic regression with its original grid, and raw/residual RBF SVM with `C=[0.1,1,10,100,1000]` and gamma factors `[0.1,1,10]/p`. Widening C follows the boundary selection observed in v1 and is explicitly a follow-up, not a correction to the original score.

## Shared compressed representation

Compare raw and residual connectivity. Within every training partition, standardise edges and fit an eight-component randomized PCA with seed 42 and five power iterations. Use its first 4, 6, or 8 components as separate settings. Standardise those components using training statistics, then map each coordinate to `pi*tanh(z/2)`. Every readout receives exactly these same bounded coordinates and the same participants.

Readouts are logistic regression, RBF SVM, polynomial SVM, and a fidelity-kernel SVM. Kernel SVMs search `C=[0.1,1,10,100]`; RBF uses gamma factors `[0.1,1,10]/d`; polynomial uses degrees 2 and 3, gamma `1/d`, and coefficient 1. Logistic regression keeps its original C grid. Dimensionality is reported separately rather than chosen from test scores. PCA is an initial classical compression control; a learned graph representation remains a distinct experiment.

The quantum feature map uses one qubit per coordinate. Each layer applies Hadamard gates to all qubits, RY rotations by the bounded coordinates, then nearest-neighbour IsingZZ rotations by the adjacent angle product divided by pi. Search depths 1 and 2 inside the inner folds. Use exact `default.qubit` statevectors with no finite-shot sampling. Compute fidelity as the squared magnitude of the inner product of the two states. Check state norms, training-kernel symmetry, unit diagonal, and positive semidefiniteness within numerical tolerance. Do not repair or centre the kernel using held-out data.

Also run the four-dimensional comparisons at training fractions 0.25 and 0.5 for both raw and residual edges. Subsample within acquisition group and diagnosis using only outer-training participants and seed 42, retaining at least two per stratum. Share that subset across all readouts and derive the inner folds from it. Keep each outer test group intact. These low-sample settings are specified before their scores are observed.

## Null and uncertainty checks

For the full-edge logistic reference, generate 99 diagnosis permutations within acquisition group, preserving site-specific class counts. Refit the full nested workflow for each permutation. Use mean site balanced accuracy and the empirical one-sided p-value `(1 + null_scores_at_least_observed)/(1 + 99)`. This tests exchangeability within the recorded groups, not clinical usefulness or the absence of nuisance effects.

For matched readouts, report paired differences by site and descriptive paired site-bootstrap intervals using common site resamples. All seven sites are the comparison set; there is no selection of a best site. Outer training sets overlap and multiple exploratory settings are tested, so a small unadjusted difference is not evidence of quantum advantage. A later independent cohort is needed to confirm a promising result.

## Validation gates

Before full runs, test residualisation on known synthetic nuisance effects and verify that perturbing held-out covariates cannot change fitted coefficients. Check PCA/scaling isolation and exact representation equality across readouts. Validate quantum fidelities against known one-qubit values and an independently implemented statevector calculation. Only then run the full cohort. Preserve v1 outputs and write each result series to a new directory.
