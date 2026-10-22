# Project log

## Purpose

This log is the durable record of why decisions were made, what changed, and what comes next. Keep task execution in GitHub Issues and summarize only material outcomes here.

## Project context

This portfolio project is designed around Mostafa Abdelmegeed's existing strengths rather than as a generic introduction to quantum machine learning:

- Senior applied AI and machine-learning experience across research and production systems.
- MSc research on brain-specific graph neural networks for Parkinson's and Alzheimer's disease classification using neuroimaging/connectivity data.
- Two peer-reviewed IEEE ISBI 2025 publications on graph learning for functional brain connectivity.
- Hands-on experience with self-supervised learning, medical imaging, distributed optimization, PyTorch, and reproducible ML systems.

The intended portfolio outcome is a rigorous hybrid quantum-classical graph-learning study. It should connect naturally to this background, include strong classical baselines, and document negative results and limitations as carefully as positive ones.

## Admissions objective

The project's ultimate purpose is to strengthen applications for graduate or serious non-degree programs in quantum machine learning, quantum computing, or closely aligned quantum information fields.

- Primary target intake: Fall 2027.
- Backup target intake: Fall 2028.
- Preferred regions: Europe, the United States, and Canada; exceptional programs elsewhere remain in scope.
- Funding preference: fully funded first, with partially funded options acceptable.
- Program-search deliverable: 15 evidence-backed options grouped as ambitious, target, and lower-risk choices. These labels describe relative admissions risk, not guaranteed outcomes.

The program search and the technical project will inform each other. Program prerequisites and research themes should shape the learning roadmap, while concrete project progress should strengthen the application narrative.

## Current focus

**Milestone 4 - Results and release**

The screened 432-participant cohort, classical reference, nuisance controls, PCA quantum kernels, trainable readouts, seed repetitions, permutation check, bounded full graph comparison, frozen graph/readout comparison, and edge co-variation experiment are complete. The latest independent audit passed for 17 series, 161 settings, and 69,552 held-out prediction rows, in addition to the permutation artifacts. The sanitized aggregate and figures are in the [results note](overnight-results.md); ABIDE II remains a metadata-only inventory until compatible quality fields are available.

### Completed

- Established the project scope: hybrid quantum-classical machine learning for neuroimaging.
- Defined the evaluation principle: quantum approaches must be assessed against matched classical baselines.
- Created the public portfolio repository.
- Selected ABIDE I with PCP C-PAC/CC200 ROI time-series derivatives as the primary dataset configuration.
- Registered a cross-site evaluation protocol with classical feature, GNN, quantum-kernel, and variational-readout models.
- Verified live access to the public metadata and a sample derivative file.
- Added a reproducible manifest generator and leakage-resistant leave-one-site-out split utility.
- Added automated tests for metadata parsing, URL safety, deterministic manifests, and site separation.
- Added a small download-to-connectivity workflow with checksum verification and an offline mode.
- Added participant-ID checks to catch site leakage even when file identifiers differ.
- Froze cohort policy v1 and the first classical evaluation before training.
- Documented dataset use terms and citation requirements.
- Audited the cohort and validated 432 participants across seven acquisition groups.
- Verified an identical offline rebuild of arrays, manifests, and exclusions.
- Ran logistic regression, RBF SVM, and a dummy control using nested acquisition-group folds.
- Verified all held-out predictions and metrics independently; all 38 automated tests pass.
- Selected PennyLane with a local simulator for the planned quantum experiments; the dependency is not installed yet.

### Next implementation tasks

1. Specify demographic/motion-only controls and a nuisance-adjusted connectivity sensitivity experiment, fitting every adjustment inside training folds.
2. Specify a wider SVM grid as a separate experiment because v1 selects the upper C boundary in six folds.
3. Implement the classical graph encoder and shared 4-8 dimensional bottleneck with matched classical readouts.
4. Add and validate the optional PennyLane environment, then freeze and run the first quantum-kernel comparison on the same representations and splits.

## Decision record

### 2026-08-01 - Project scope

The project will connect quantum machine learning to neuroimaging and graph-learning experience, rather than reproduce a generic quantum tutorial. The first experiments should remain small enough to run reproducibly on simulators.

### 2026-08-01 - Tracking approach

GitHub is the source of truth. The README exposes the public roadmap, GitHub Issues represent actionable work, and this log captures durable context. A separate Notion workspace is optional and will be added only if it becomes useful for personal reading notes.

### 2026-08-01 - First dataset and evaluation design

ABIDE I was selected because it is public, graph-native after functional-connectivity construction, and multi-site. The primary evaluation will hold out acquisition sites so that scanner/site signal cannot be mistaken for disease generalization. The first quantum models will operate on low-dimensional graph representations; raw fMRI and full 200-node graphs will not be loaded directly into a quantum circuit.

### 2026-10-22 - Connectivity smoke test

Resumed development and added a reproducible download-to-connectivity command. The default integration cohort contains eight participants from NYU and PITT, balanced within site and diagnosis, with PITT held out. The command checks download lengths and local SHA-256 receipts, validates ROI order and signals, and writes Pearson matrices, Fisher-z matrices, and aligned edge features. A cached offline run also completed.

The full manifest still contains 1,035 participants across 20 sites and reproduces the original fingerprint. All 28 automated tests pass. The tests cover numerical edge cases, deterministic cohort selection, damaged caches, truncated downloads, offline reconstruction, and participant separation across sites.

The environment uses Python 3.11 and locked dependencies. Full research inclusion rules remain open; the next step is a cohort quality audit and a written exclusion policy before training classical baselines.

### 2026-10-22 - Cohort freeze and first classical reference

Froze [cohort policy v1](cohort-policy.md) using the PCP motion cutoff, explicit OK ratings from all five manual QC fields, complete age/sex/motion metadata, a 100-timepoint minimum, and minimum acquisition-group counts. Leuven, UCLA, and Michigan's paired samples stay together in every split. [Dataset use and citations](data-use.md) are documented.

Metadata screening retained 434 participants. Two NYU control scans had constant ROI signals, leaving 432 participants across seven acquisition groups. A second build using only cached files reproduced every array, the manifest, and all exclusion decisions. The [cohort audit](cohort-audit.md) records attrition and demographic differences.

The first nested evaluation produced mean balanced accuracy of 69.06% for logistic regression and 69.05% for RBF SVM, with mean AUROC 0.781 and 0.780. The dummy control scored 50% and 0.5 per site. [Full results](classical-results.md) include site variation, descriptive intervals, calibration, and tuning limitations. Motion and sex differences persist after QC; the next controls must assess them before interpreting these scores as disease-related connectivity effects.

All 38 automated tests pass. A separate prediction audit reproduced each site's metrics from saved predictions and confirmed participant and group separation. The [quantum software decision](quantum-stack.md) selects PennyLane for the next stage; no quantum model has been trained.

### 2026-10-22 - Matched quantum and nuisance follow-ups

The original classical run was rebuilt and rerun offline. Its arrays, manifest, exclusion decisions, and 1,296 held-out predictions matched the preserved run; the prediction CSV is byte-identical. The hardened workflow records source, environment, configuration, and input receipts, and rejects changes during a run.

The full-edge logistic reference averaged 69.1% balanced accuracy and 0.781 AUROC across the seven acquisition groups. Age, sex, and motion alone reached 56.3% and 0.625. Fold-fitted nuisance residualization reduced logistic performance to 64.5% and 0.713. This makes nuisance handling central to interpretation, without establishing a causal explanation for the score change.

The four-coordinate fidelity quantum kernel on raw full-data PCA reached about 64.6% balanced accuracy across seeds 42, 137, and 2026. Its mean paired difference was 4.95 points against logistic and 5.70 against RBF SVM; the descriptive seven-site interval against logistic reaches zero, while the interval against RBF remains positive. The signal did not hold at six or eight coordinates or at half and quarter training fractions. Residualized PCA narrowed the difference to classical models.

The trainable four-qubit readout did not consistently match its 19-parameter MLP comparator. A 99-permutation within-site label check placed the logistic reference above all shuffled statistics (empirical one-sided value 0.01). The artifact auditor independently checks the observed predictions and permutation statistic arithmetic, but does not refit every null model.

An ABIDE II metadata-only inventory and offline reproduction both returned 1,007 candidates from 16 readable tables. The inventory lacks the motion and manual QC fields required by the primary cohort policy; no external imaging model was run. One polynomial SVM fit in the first full graph-kernel run remained inside an inner fold for more than nine hours. That partial run is retained as failed; the complete rerun used a fixed solver cap and a copied encoder cache checked by a saved hash manifest. The frozen-encoder graph/readout experiment is complete and is covered by the final audit. See the [current results note](overnight-results.md), [external readiness review](external-validation-readiness.md), and [reproduction steps](reproduction.md).

The final independent audit covers 17 model series, 161 settings, and 69,552 held-out prediction rows, plus the permutation artifacts. All 91 automated tests pass. The aggregate and figures are exported and linked from the results note. The graph readout was below the matched MLP on raw and residual embeddings; the raw PCA4 quantum kernel remains the main exploratory quantum result. The new 16-module classical representation reached 65.3% balanced accuracy, but its modules were unstable across outer folds and it did not outperform the full-edge logistic reference.


### 2026-10-22 - Higher-order edge co-variation representation

Tested a second layer of organization over connectivity edges. In each nested training split, the model selects 256 high-variance edges, clusters their across-participant correlations into 4, 8, or 16 signed modules, and summarizes participants by module. The groups are learned without labels. The 16-module raw logistic model averaged 65.3% balanced accuracy (0.668 AUROC), below the full-edge logistic reference at 69.1%. The matched four-module quantum kernels scored near chance; depth 2 trailed matched logistic by 4.27 points on raw data (descriptive seven-site interval -8.28 to -0.09 points).

A fold-overlap diagnostic found modest module stability: selected-edge Jaccard overlap was 58% to 62%; signed module overlap after matching was 30% to 42%. This limits biological interpretation and motivates independent confirmation with a frozen grouping rule before further optimization. The [protocol](covariation-protocol.md), [results and figures](overnight-results.md), and [reproduction instructions](reproduction.md) record the experiment. The final audit now covers 17 series, 161 settings, and 69,552 predictions, plus the permutation artifacts.
