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

## Initial research direction

Start with a small, public neuroimaging-derived classification dataset or a carefully prepared connectivity-feature benchmark. The first decision will optimize for reproducibility and ethical data access, not for benchmark size.

## Repository structure

```text
docs/        Project log, literature notes, and decision records
notebooks/   Exploratory and reproducible experiments
src/         Reusable data, modeling, and evaluation code
tests/       Automated checks as the codebase matures
```

