# ABIDE I data card

Last verified: 2026-08-01

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

## Current inclusion state

The verified counts above apply only the availability filter: a row must have a valid PCP `FILE_ID`. Motion, imaging quality, missing covariates, site sample size, and other scientific exclusion rules have not yet been applied. Those rules must be frozen before full experiments.

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

