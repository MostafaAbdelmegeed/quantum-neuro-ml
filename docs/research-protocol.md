# Milestone 0 research protocol

Status: working protocol to be frozen before model implementation

## Working title

**Hybrid Quantum-Classical Graph Learning for Cross-Site Classification of Functional Brain Connectomes**

## Research question

Under a leakage-resistant, low-sample evaluation protocol, can quantum kernels or a variational quantum readout match or improve parameter-matched classical readouts applied to learned representations of functional brain-connectivity graphs?

This project does not begin with a claim of quantum advantage. It asks a falsifiable comparative question and treats a well-supported negative result as a valid outcome.

## Why this is a useful portfolio and research project

- It builds directly on prior graph-neural-network and neuroimaging research experience.
- It requires genuine quantum-computing knowledge rather than adding a decorative quantum layer to an image classifier.
- It tests an important practical regime for QML: small samples, high-dimensional classical data, and limited qubits.
- It can produce a reproducible benchmark even if the quantum models do not outperform the classical models.
- It creates a coherent application narrative linking past graph-learning work to future QML study.

## Dataset decision

### Primary dataset: ABIDE I

ABIDE I aggregates resting-state fMRI, anatomical imaging, and phenotypic data from multiple international sites for autism-spectrum-disorder research. The Preprocessed Connectomes Project (PCP) provides openly shared derivatives, including ROI time series from several atlases and multiple preprocessing pipelines.

Primary working configuration:

- PCP C-PAC pipeline.
- Band-pass filtered, without global signal regression as the primary strategy.
- Craddock 200 ROI time series as the primary atlas.
- Autism spectrum disorder versus typical control as the initial binary target.
- A global-signal-regression variant as a sensitivity analysis, not as an extra model-selection degree of freedom.

Why ABIDE I:

- Public, documented, and widely used enough to support reproducibility.
- Functional connectivity maps naturally to a graph representation.
- Multi-site acquisition creates a meaningful out-of-distribution generalization test.
- Preprocessed ROI time series avoid making full fMRI preprocessing the first bottleneck.

Before downloading data, record the applicable data-use terms, citation requirements, subject inclusion rules, and a checksumed manifest.

## Graph construction

For each participant:

1. Compute the ROI-by-ROI Pearson correlation matrix from the ROI time series.
2. Apply Fisher's z transform to off-diagonal correlations.
3. Construct a weighted undirected graph with ROIs as nodes and functional connectivity as edges.
4. Derive thresholds only from training data. Compare a fixed-density graph with a weighted graph in a predeclared sensitivity analysis.
5. Use reproducible node features, initially connectivity profiles plus low-dimensional spectral positional encodings.

All nuisance handling, scaling, feature selection, dimensionality reduction, and threshold selection must be fitted inside the training fold.

## Models

### Non-graph classical baselines

1. Regularized logistic regression on vectorized upper-triangle connectivity features.
2. RBF SVM on the same features.
3. A tangent-space-connectivity baseline if it can be implemented without leakage.

### Classical graph baselines

1. GCN encoder plus MLP readout.
2. GIN encoder plus MLP readout.
3. A parameter-matched MLP readout on exactly the embedding used by the quantum model.

### Quantum models

1. **Quantum-kernel SVM:** reduce a training-only graph representation to 4-8 dimensions, angle encode it, and compare a fidelity-style quantum kernel with RBF and polynomial kernels.
2. **Hybrid variational readout:** map a 4-8 dimensional GNN graph embedding to a parameterized quantum circuit and use measured expectations for binary classification.

Simulator experiments come first. A small run on real quantum hardware is optional and should test noise sensitivity, not be presented as proof of practical advantage.

## Fair comparison rules

- Use identical outer test splits across all models.
- Tune every model only within the corresponding training data.
- Match the quantum and classical readouts by input embedding, data split, and approximately by trainable-parameter budget.
- Report runtime, circuit depth, qubit count, number of shots, and simulator/hardware details.
- Include a classical model with the same dimensional bottleneck used for the quantum circuit.
- Include a random or untrained quantum-feature control where appropriate.
- Do not claim quantum advantage from a small mean-score difference.

## Evaluation protocol

### Primary evaluation

Leave-one-site-out or grouped cross-site evaluation, subject to a minimum usable sample count per held-out site. This measures generalization to an unseen acquisition site.

### Secondary evaluation

Repeated site-stratified nested cross-validation for a lower-variance in-distribution estimate and for controlled sample-size experiments.

### Primary metrics

- Balanced accuracy.
- AUROC.

### Secondary metrics

- Macro F1.
- Matthews correlation coefficient.
- Brier score and calibration error.
- Runtime and model/circuit complexity.

### Uncertainty and statistical analysis

- Report confidence intervals across held-out sites and repeated splits.
- Use paired bootstrap or another paired resampling method on identical predictions/splits.
- Use permutation testing for the central performance comparison if computationally feasible.
- Report effect sizes and the distribution of results, not only p-values.

## Core hypotheses

- **H1:** A hybrid quantum readout can match a parameter-matched classical readout when both consume the same low-dimensional graph embedding.
- **H2:** Quantum kernels may behave differently from classical kernels in the smallest training-set regimes, but any gain must persist across sites and repeated splits.
- **H3:** Site shift will be a larger source of performance variation than the choice between a small quantum and classical readout.

H1 and H2 are empirical questions, not expected conclusions. H3 is included to keep the medical-ML interpretation honest.

## Ablations

- 4 versus 6 versus 8 qubits where computationally feasible.
- Quantum feature map and circuit depth.
- Entanglement topology.
- Exact simulation versus finite-shot simulation.
- Frozen versus jointly trained classical graph encoder.
- Weighted versus fixed-density graphs.
- With versus without global signal regression as a predeclared preprocessing sensitivity analysis.
- Full-data versus controlled low-sample training subsets.

## Scope control

Not in the first paper:

- Raw-volume fMRI deep learning.
- Multiple diseases or multiple datasets.
- Claims of clinical utility.
- Claims of quantum computational advantage.
- Large real-hardware training runs.
- A new fMRI preprocessing pipeline.

## Publication path

1. Reproducible technical report in the repository.
2. Preprint after the protocol, code, and full results are stable.
3. Submission to a suitable QML, graph-learning, or neuroimaging venue after a separate venue review.

Independent submission is possible. A research collaborator may be approached after the protocol and initial baselines exist, but authorship must reflect a real intellectual or experimental contribution.

## Milestone 0 exit criteria

Milestone 0 is complete when:

- Dataset access and citation terms are documented.
- A frozen subject-inclusion and preprocessing configuration exists.
- The data manifest can be generated reproducibly.
- The exact split logic and leakage tests are specified.
- Baselines, quantum models, metrics, and ablations are registered in this document.
- A small smoke-test subset can run end to end without using test data for fitting.

## Initial sources

- [ABIDE I overview](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
- [ABIDE preprocessed derivatives](https://preprocessed-connectomes-project.org/abide/derivatives.html)
- [ABIDE preprocessed download structure](https://preprocessed-connectomes-project.org/abide/download.html)
- [PCP preprocessing pipelines](https://preprocessed-connectomes-project.org/abide/Pipelines.html)
- [Towards Quantum Graph Neural Networks: An Ego-Graph Learning Approach](https://arxiv.org/abs/2201.05158)
- [Hybrid Quantum-Classical Graph Convolutional Network](https://arxiv.org/abs/2101.06189)
- [Hybrid Quantum Graph Neural Network for Molecular Property Prediction](https://arxiv.org/abs/2405.05205)

