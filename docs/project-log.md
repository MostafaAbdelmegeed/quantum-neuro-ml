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

**Milestone 0 - Foundations**

### Completed

- Established the project scope: hybrid quantum-classical machine learning for neuroimaging.
- Defined the evaluation principle: quantum approaches must be assessed against matched classical baselines.
- Created the public portfolio repository.
- Selected ABIDE I with PCP C-PAC/CC200 ROI time-series derivatives as the primary dataset configuration.
- Registered a cross-site evaluation protocol with classical feature, GNN, quantum-kernel, and variational-readout models.
- Verified live access to the public metadata and a sample derivative file.
- Added a reproducible manifest generator and leakage-resistant leave-one-site-out split utility.
- Added automated tests for metadata parsing, URL safety, deterministic manifests, and site separation.

### Next implementation tasks

1. Freeze subject inclusion, imaging quality, motion, and minimum site-size rules.
2. Implement derivative download verification and checksum capture for a small smoke-test cohort.
3. Construct functional-connectivity matrices without fitting any transform on held-out sites.
4. Select the initial quantum software stack after the classical data contract is stable.

## Decision record

### 2026-08-01 - Project scope

The project will connect quantum machine learning to neuroimaging and graph-learning experience, rather than reproduce a generic quantum tutorial. The first experiments should remain small enough to run reproducibly on simulators.

### 2026-08-01 - Tracking approach

GitHub is the source of truth. The README exposes the public roadmap, GitHub Issues represent actionable work, and this log captures durable context. A separate Notion workspace is optional and will be added only if it becomes useful for personal reading notes.

### 2026-08-01 - First dataset and evaluation design

ABIDE I was selected because it is public, graph-native after functional-connectivity construction, and multi-site. The primary evaluation will hold out acquisition sites so that scanner/site signal cannot be mistaken for disease generalization. The first quantum models will operate on low-dimensional graph representations; raw fMRI and full 200-node graphs will not be loaded directly into a quantum circuit.
