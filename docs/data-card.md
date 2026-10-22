# ABIDE I data card

Last verified: 2026-10-22

## Intended use

The first experiment uses ABIDE I resting-state fMRI ROI time series to construct subject-level functional-connectivity graphs for autism-versus-control classification. The study is methodological and does not claim clinical validity.

## Source

- Phenotypic metadata: `Phenotypic_V1_0b_preprocessed1.csv` from the public ABIDE Initiative S3 storage.
- Derivative configuration: C-PAC / band-pass filtered / no global signal regression / Craddock 200 ROI time series.
- Derivative format: one `.1D` ROI time-series file per downloadable subject.

Official documentation:

- [ABIDE I overview](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
- [Preprocessed derivatives](https://preprocessed-connectomes-project.org/abide/derivatives.html)
- [Download URL structure](https://preprocessed-connectomes-project.org/abide/download.html)
- [Preprocessing pipelines and strategies](https://preprocessed-connectomes-project.org/abide/Pipelines.html)

## Verified manifest summary

The local manifest generator produced the following result from the source metadata:

- Published metadata rows: 1,112.
- Rows with downloadable derivative identifiers: 1,035.
- Acquisition sites represented: 20.
- Autism group: 505.
- Typical-control group: 530.
- Generated manifest SHA-256: `5bb56fcce8082a3a4aa91fc75544977cc51463913744aacc1897fbdea3dfbd43`.

This fingerprint covers the generated minimal manifest, not the complete set of derivative files.

The 2026-10-22 run reproduced these counts and the same manifest fingerprint. The cached source metadata has SHA-256 `009f01a89697113e5375ded4fbcff4c85e89a4daa6d13e63c21ae5d57dda0865`.

## Connectivity integration check

The [smoke-test workflow](smoke-test.md) completed on eight participants from NYU and PITT, with two participants per diagnosis at each site. PITT supplied four test participants; NYU supplied four training participants.

Every selected scan had 200 ROI columns and either 176 or 196 timepoints. The output contains eight finite, symmetric 200-by-200 Pearson matrices and corresponding Fisher-z matrices, with 19,900 upper-triangle edges per participant. Pearson diagonals are one; Fisher-z diagonals are zero. The offline rebuild completed using the cached files and checksum receipts.

This check used Python 3.11.15 and NumPy 2.4.6. It verifies the integration path only. It does not apply research motion or imaging-quality exclusions and does not establish a usable research cohort.

## Current inclusion state

The 1,035-participant inventory above applies the derivative-availability filter. [Cohort policy v1](cohort-policy.md), frozen before training on 2026-10-22, adds motion, manual imaging quality, required covariates, scan validation, and acquisition-group minimums. [Data-use and citation requirements](data-use.md) are documented separately.

Metadata screening retains 434 participants (205 autism, 229 control) across seven acquisition groups. Scan validation excludes two NYU controls with constant ROI signals, leaving 432 participants (205 autism, 227 control). All seven groups remain eligible. The [cohort audit](cohort-audit.md) records attrition, demographic summaries, and final fingerprints. A separate offline rebuild produced identical arrays and manifests.

## Privacy and repository policy

- Imaging data, ROI time series, and generated subject manifests are not committed to Git.
- Only aggregate counts, source documentation, code, and non-sensitive fingerprints are versioned.
- No attempt will be made to identify participants.
- The project must preserve all dataset citation, acknowledgement, and data-use requirements in any report or publication.

## Known risks

- Site and scanner effects may dominate disease-related signal.
- Preprocessing strategy, atlas choice, motion, age, and sex can materially affect results.
- Random subject splits can overestimate generalization when sites occur in both train and test data.
- Autism is heterogeneous, and binary classification is a simplified methodological target rather than a diagnostic claim.

