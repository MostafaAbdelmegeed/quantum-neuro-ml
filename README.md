# Quantum Neuro ML

A long-term, reproducible portfolio project exploring hybrid quantum-classical machine learning for neuroimaging.

## Goal

Evaluate whether quantum feature maps and variational quantum classifiers add measurable value over carefully tuned classical baselines for a constrained neuroimaging classification task.

## Guiding principle

This is an evidence-first project. Every quantum result will be compared against a matched classical baseline, using the same split, preprocessing, metrics, and compute-budget notes.

## Roadmap

- [ ] **Milestone 0 - Foundations:** establish environment, reading list, data decision, and reproducibility rules.
- [ ] **Milestone 1 - Classical baseline:** implement a compact, well-evaluated baseline on a selected public neuroimaging-derived dataset.
- [ ] **Milestone 2 - Quantum kernel:** test a quantum feature map/kernel under the same evaluation protocol.
- [ ] **Milestone 3 - Variational classifier:** compare a hybrid variational circuit against the baselines.
- [ ] **Milestone 4 - Portfolio release:** publish results, limitations, reproducibility instructions, and a concise technical write-up.

## Project tracking

Progress and decisions are maintained in [the project log](docs/project-log.md). GitHub Issues will track discrete tasks; this document records decisions and research notes that need context.

The current experimental design is defined in the [Milestone 0 research protocol](docs/research-protocol.md). It must be frozen before full model implementation so that dataset and evaluation choices are not changed after seeing results.

## Initial research direction

Start with the public, multi-site ABIDE I resting-state fMRI dataset and PCP ROI time-series derivatives. The initial task is cross-site autism-versus-control classification from functional-connectivity graphs, with classical and hybrid quantum-classical models evaluated under identical leakage-resistant splits.

## Repository structure

```text
docs/        Project log, literature notes, and decision records
notebooks/   Exploratory and reproducible experiments
src/         Reusable data, modeling, and evaluation code
tests/       Automated checks as the codebase matures
```
