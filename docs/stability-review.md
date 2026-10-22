# Foundation stability review

Review date: 2026-10-22. The reviewed state was the working tree against commit `72ad025`, including untracked implementation files. The local research protocol and cohort policy supplied the specification. The original v1 code, lockfile, reports, and predictions were preserved before fixes in `artifacts/checkpoints/classical-v1-original.zip`, SHA-256 `80f0e405fbb8c155ea0d75eb83c1962799d1d9894d4ca4f933d6f8a84e5b168c`.

## Standards

No hard repository-rule violations were found. Two reproducibility weaknesses needed correction:

- Source and lockfile fingerprints were captured after evaluation. An edit during training could therefore identify source bytes that were not executed. Dataset builds also lacked a source snapshot.
- The dataset loader verified the archive and policy hashes but trusted other lineage fields from the report. Replacing its metadata and manifest fingerprints in memory with zeroes still allowed the dataset to load.

Runs now capture source, environment, configuration, and input receipts before computation. They preserve a source archive, reject changes to captured source or inputs before completion, and record running, failed, or complete status. The loader now checks metadata identity, derivative configuration, manifest digest, participant/label/site/covariate alignment, and covariate inclusion rules.

## Spec

The review found no evidence invalidating the v1 results. All 21 model/group metric records were independently reconstructed from saved predictions. Identities, diagnoses, groups, covariates, and recorded fingerprints matched their artifacts. Scaling and parameter selection stayed inside training partitions, and paired samples from the same acquisition centre remained together.

The complete-lineage enforcement issue also affected the specification: the metadata snapshot and C-PAC derivative configuration were documented requirements but were not fully checked at load time. New rejection tests exercise those failures and malformed nuisance covariates.

Graph encoders, shared bottlenecks, quantum models, secondary evaluation, and paired comparisons were correctly disclosed as unfinished. No additional missing requirement blocks nuisance controls.

Standards findings: two reproducibility issues. Spec findings: one lineage-enforcement issue, shared with the standards review. These fixes do not change cohort selection, v1 folds, or v1 model grids. Later experiments are exploratory follow-ups to an already observed reference run, not a fresh blinded validation study.

## Verification after fixes

An offline rebuild in `data/cohort/v1-verified/` reproduced all cohort arrays, participant decisions, and manifest bytes. The resulting classical rerun in `results/classical-v1-verified/` produced byte-identical predictions and exactly equal summary metrics to the preserved original run. The foundation suite passed all 66 tests before the v2 experiments started.

The v2 review checked nuisance residualisation, PCA isolation, matched representation fingerprints, and within-group label permutations. A synthetic Windows process-pool trial completed successfully. Full-cohort permutation execution also recomputes the original observed logistic statistic and checks every held-out score before starting null trials.

Each result directory keeps its own source archive and completion receipt. Later source changes do not rewrite earlier runs; reproduce an earlier result using its archived source and lockfile. Archives also contain input manifests and reports and must remain with the ignored local participant artifacts.
