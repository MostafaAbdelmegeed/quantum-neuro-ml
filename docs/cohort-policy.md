# Cohort policy v1

Frozen on 2026-10-22, before the first model run. The executable configuration is [cohort-v1.json](../config/cohort-v1.json). Each audit, dataset, and model report records its canonical SHA-256. A change to any inclusion threshold requires a new policy version and a separate result series.

## Inclusion rules

Use the cached PCP metadata snapshot with SHA-256 `009f01a89697113e5375ded4fbcff4c85e89a4daa6d13e63c21ae5d57dda0865`. Require a downloadable file identifier, a unique participant ID, a site label, and diagnosis code 1 or 2. Duplicate participant or file identifiers stop the audit.

Require finite positive age, published sex code 1 or 2, and finite mean framewise displacement (FD) in `[0, 0.2)` mm. Empty values, non-finite numbers, and the `-9999` sentinel count as missing. Keep all available ages and both published sex categories; do not select by IQ, handedness, medication, or diagnosis severity. Age, sex, and FD support cohort descriptions and later nuisance analyses. They are not classifier inputs in the first connectivity baseline.

The motion threshold follows the [PCP download script](https://github.com/preprocessed-connectomes-project/abide/blob/fda1ec5d3f53eb97d94352e779f37c3ce46caee1/download_abide_preproc.py), which excludes mean FD at or above 0.2 mm. This does not mean every retained frame has FD below 0.2 mm. This workflow does not censor individual frames.

Require `OK` in all five manual rating columns: `qc_rater_1`, `qc_anat_rater_2`, `qc_func_rater_2`, `qc_anat_rater_3`, and `qc_func_rater_3`. Exclude `maybe`, `fail`, missing, or unrecognised ratings. PCP describes rater 1 as inspecting preprocessed functional data and derivatives, and raters 2 and 3 as inspecting raw anatomy and functional data. This all-rater rule is a conservative project choice, not a PCP recommendation. It may change the demographic composition and reduce generalisability. See the [official QC documentation source](https://github.com/preprocessed-connectomes-project/abide/blob/gh-pages/quality_assessment.md).

## Acquisition groups and scan validation

Keep `LEUVEN_1` and `LEUVEN_2` together as `LEUVEN`, `UCLA_1` and `UCLA_2` as `UCLA`, and `UM_1` and `UM_2` as `UM`. Retain original site labels in local manifests. ABIDE I comprises 17 contributing sites; its metadata separates some sites into multiple samples. These conservative groups prevent a second sample from the same centre appearing in training when that centre is held out. See the [ABIDE I overview and site profiles](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html).

An acquisition group must retain at least 20 participants, including at least 10 in each diagnosis group. These are practical minimums for estimating balanced accuracy and AUROC by site. They are fixed before performance is observed and do not imply adequate statistical power. Apply them after metadata screening and again after validating the downloaded scans.

Use C-PAC / `filt_noglobal` / `rois_cc200` only. Each scan must contain ordered ROI columns 1 through 200, finite nonconstant signals, and at least 100 timepoints. The 100-volume minimum is a project choice to reject very short series; different repetition times mean it is not a uniform duration in seconds. Preserve signed Pearson edges, clip correlations by `1e-7` for Fisher transformation, set the Fisher diagonal to zero, and extract the 19,900 upper-triangle edges. Do not threshold, impute, or discard ROI columns.

Invalid scans receive an explicit exclusion reason. Network errors, missing downloads, and cache-integrity failures stop the build so that transport failures cannot silently redefine the cohort. A rerun reuses verified downloads. No participant is substituted for an excluded scan.

## Audit and evaluation

The audit lists all reasons per participant locally and publishes aggregate counts only. Its first-reason counts use this order: missing derivative; invalid site or diagnosis; missing covariates; missing motion; excessive motion; manual QC; site minimum. Counts before scan validation are metadata eligibility counts, not a verified imaging cohort.

The first classical evaluation uses outer leave-one-acquisition-group-out folds and three inner `GroupKFold` folds on the remaining groups. Every inner training and validation partition must contain both diagnoses. The complete scaling/classification pipeline is fitted separately inside each inner fold. Select parameters by mean inner balanced accuracy, refit on the outer training data, and score the held-out group once.

The initial models are class-balanced L2 logistic regression and class-balanced RBF SVM, plus a training-prior dummy control. Use all edges and `StandardScaler`; do not fit feature selection, PCA, nuisance regression, or site harmonisation in this first reference run. Fix seed 42. Logistic regression searches `C = [0.001, 0.01, 0.1, 1]`; SVM searches `C = [0.1, 1, 10]` and `gamma = [0.1/p, 1/p, 10/p]`, where `p` is the edge count. These grids are fixed before model results.

Report balanced accuracy and AUROC per held-out group, their unweighted means across groups, and pooled out-of-fold metrics separately. Include macro F1, Matthews correlation, runtime, and selected parameters. Report Brier score and 10-bin expected calibration error for logistic regression and the dummy control. SVM uses its decision score for AUROC and has no probability calibration in this run. It therefore has no Brier or calibration score. Site-bootstrap intervals describe variation across these observed sites; they are not independent-cohort validation or a test of quantum advantage.

The graph encoders, dimensional bottleneck, quantum models, preprocessing sensitivities, repeated secondary evaluation, and paired comparisons remain later experiments. Their implementations must follow the wider [research protocol](research-protocol.md).
