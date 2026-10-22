# Quantum Neuro ML

A long-term, reproducible portfolio project exploring hybrid quantum-classical machine learning for neuroimaging.

## Goal

Evaluate whether quantum feature maps and variational quantum classifiers add measurable value over carefully tuned classical baselines for a constrained neuroimaging classification task.

## Guiding principle

This is an evidence-first project. Every quantum result will be compared against a matched classical baseline, using the same split, preprocessing, metrics, and compute-budget notes.

## Roadmap

- [x] **Milestone 0 - Foundations:** establish environment, reading list, data decision, and reproducibility rules.
- [ ] **Milestone 1 - Classical baseline:** implement a compact, well-evaluated baseline on a selected public neuroimaging-derived dataset.
- [ ] **Milestone 2 - Quantum kernel:** test a quantum feature map/kernel under the same evaluation protocol.
- [ ] **Milestone 3 - Variational classifier:** compare a hybrid variational circuit against the baselines.
- [ ] **Milestone 4 - Portfolio release:** publish results, limitations, reproducibility instructions, and a concise technical write-up.

## Project tracking

Progress and decisions are maintained in [the project log](docs/project-log.md). GitHub Issues will track discrete tasks; this document records decisions and research notes that need context.

The [research protocol](docs/research-protocol.md) defines the study. [Cohort policy v1](docs/cohort-policy.md) freezes the data rules and first classical evaluation before model training.

The [first classical results](docs/classical-results.md) use 432 validated participants across seven acquisition groups. Logistic regression and RBF SVM both average about 69.1% balanced accuracy and 0.78 AUROC. Nuisance controls and graph/quantum models remain next.

## Initial research direction

Start with the public, multi-site ABIDE I resting-state fMRI dataset and PCP ROI time-series derivatives. The initial task is cross-site autism-versus-control classification from functional-connectivity graphs, with classical and hybrid quantum-classical models evaluated under identical leakage-resistant splits.

## Repository structure

```text
docs/        Project log, literature notes, and decision records
config/      Versioned cohort inclusion policies
src/         Reusable data, modeling, and evaluation code
tests/       Automated checks as the codebase matures
```

## Development quick start

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). The recorded development environment uses Python 3.11; `uv.lock` fixes dependency versions.

```bash
uv sync --locked --extra dev --python 3.11
uv run --locked --extra dev python -m pytest
```

Generate the local ABIDE derivative manifest without downloading the imaging data:

```bash
uv run --locked --extra dev qneuro-abide-manifest --output data/manifests/abide-cpac-filt-noglobal-cc200.csv
```

Generated subject manifests and downloaded data remain untracked. Verified cohort counts and the source-metadata fingerprint are recorded in [the ABIDE data card](docs/data-card.md).

Run the connectivity smoke test on eight participants from NYU and PITT:

```bash
uv run --locked --extra dev qneuro-abide-smoke
uv run --locked --extra dev qneuro-abide-smoke --offline
```

The command downloads ROI time series, records file checksums, and builds Pearson and Fisher-z connectivity matrices. It reserves PITT as the test site. Outputs stay in the ignored `data/` directory. See [the smoke-test contract](docs/smoke-test.md) for selection rules and output formats. This cohort checks the pipeline independently of the research inclusion rules.

Build the screened research cohort and run the classical reference experiment:

```bash
uv run --locked --extra dev qneuro-abide-cohort
uv run --locked --extra dev qneuro-abide-build --workers 4
uv run --locked --extra dev qneuro-abide-baseline
```

These commands use the pinned metadata snapshot described in [the reproduction instructions](docs/classical-baselines.md). The models use nested acquisition-group folds, with scaling fitted inside each training fold. Participant data and predictions stay in ignored directories; aggregate findings belong in `docs/`.
