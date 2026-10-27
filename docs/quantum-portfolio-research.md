# Quantum portfolio project candidates

Research checked on 23 October 2026. The aim is a portfolio of reproducible quantum ML and Qiskit projects. A useful result can be a careful reproduction, a better implementation, an informative comparison, or a measured failure mode. Quantum advantage is a separate claim and is not an acceptance criterion.

## Recommendation

Start with a quantum generative learning project in Qiskit. First reproduce a six-qubit quantum circuit Born machine (QCBM), a trainable circuit that generates small binary patterns. Then extend the project toward IQP generative models and study how hardware connectivity and noise change the generated distribution. This adds generative modeling to the classification work already in this repository.

For portfolio breadth, follow this with ML-guided Qiskit compilation, then warm-start QAOA. The first teaches quantum ML, the second produces a quantum software tool, and the third demonstrates quantum optimization. QCNN phase classification and quantum autoencoders are alternatives if the preference is to concentrate on quantum-state learning. The completed neuroimaging study remains an existing portfolio entry.

These are portfolio recommendations, not predictions of a quantum performance advantage. The proposed extensions below are experiment ideas. This search does not establish that they are new to the literature.

| Candidate | Initial scale | Portfolio output | Main cost or complication |
| --- | --- | --- | --- |
| QCBM generative learning | 4-8 qubits | Generated patterns, distribution diagnostics, training and shot-cost curves | Optimizing a sampled distribution; choosing a loss that detects the correlations of interest |
| ML-guided Qiskit compilation | Small circuit families; larger circuits need no statevector simulation for compilation | A compiler-configuration selector and cost/quality benchmark | Training-data generation, circuit-family leakage, and testing under changed device assumptions |
| Warm-start QAOA | 6-12 decision qubits | Optimization benchmark with exact references and a sampling demo | Fair accounting of classical initialization and quantum optimization costs |
| QCNN phase classification | 4 and 8 qubits | Phase diagram, measurement-budget comparison, noisy inference results | Ground-state preparation and fair access to input states |
| Quantum autoencoder | 4-8 input qubits, 2-4 latent qubits | Compression curve, anomaly score, fault-specific detection results | Compression fidelity does not guarantee detection of every anomaly |

## 1. QCBM: learning to generate structured bitstrings

Liu and Wang's *Differentiable learning of quantum circuit Born machines* trains a parameterized circuit using maximum mean discrepancy (MMD), a discrepancy between samples from the target and model distributions. Measurement produces a bitstring drawn from the circuit's learned distribution. The paper studies Bars-and-Stripes and Gaussian mixtures. [Original paper, Physical Review A 98, 062324 (2018)](https://arxiv.org/abs/1804.04168)

The authors provide a six-bit Gaussian notebook, Bars-and-Stripes experiments, and a Python implementation. Their code uses an older Python/ProjectQ stack, so a current Qiskit implementation would be a port and reproduction rather than an unchanged execution of the original software. The repository has an MIT license. [Author repository](https://github.com/GiggleLiu/QuantumCircuitBornMachine) [License](https://github.com/GiggleLiu/QuantumCircuitBornMachine/blob/master/LICENSE)

### A bounded first experiment

Use six qubits for a discretized Gaussian mixture and 2-by-3 Bars-and-Stripes images. Generate training and held-out samples with recorded seeds. Build shallow rotation-and-entangler circuits with Qiskit, evaluate exact probabilities locally, and train with MMD. The circuit starts from zero; each measurement generates a new example. There is no input vector to amplitude-encode for every training example.

The first extension should compare a fixed-bandwidth MMD loss with a multiscale MMD loss under an equal total circuit-evaluation and shot budget. Keep the circuit fixed during this comparison. Add an even-parity distribution as a diagnostic after the initial reproduction: it tests whether a model captures a correlation that is absent from lower-order marginals. Existing work studies both multiscale kernels and bandwidth adaptation alongside the tradeoff between trainability and high-order correlations. The contribution here would be the reproducible resource comparison, not the kernel idea. [Rudolph et al., npj Quantum Information 10, 116 (2024)](https://www.nature.com/articles/s41534-024-00902-0)

Compare against an empirical multinomial distribution, independent Bernoulli variables, a pairwise model, and a small autoregressive classical network. The empirical distribution is a serious baseline at this small support size. Give every learner the same training samples; keep the true target probabilities for evaluation only.

Report held-out MMD, Jensen-Shannon divergence and total variation on the small enumerated support, invalid-pattern rate, coverage of valid patterns, and selected higher-order correlations. Include training time, objective evaluations, shots, two-qubit gates and depth after transpilation. Show all seeds. A good plot is generated-pattern quality against cumulative shots, with the classical baselines beside it.

### Qiskit and hardware

Use `QuantumCircuit`, parameter binding, and `Statevector` for the exact reference. Use a V2 sampler for finite-shot runs. `SamplerQNN` is available if a neural-network interface is useful, but a direct sampler loop is enough for a Born machine; a discriminator or qGAN framework is unnecessary. The current interface accepts V2 primitives and can return the full bitstring probability vector. [SamplerQNN API](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.SamplerQNN.html)

The initial project needs a CPU and a local simulator. A hardware follow-up can evaluate a few frozen six-qubit circuits. Full hardware training should wait until a shot ledger and transpiled circuit costs are available. No speedup claim follows from reproducing these small distributions.

### A more recent research project to extend

Recio-Armengol, Ahmed and Bowles propose training instantaneous quantum polynomial (IQP) generative circuits through classically estimated objectives, then using quantum circuits to draw samples. Their preprint was submitted in 2025 and revised in February 2026. The authors provide experiment code, dataset scripts and trained-model instructions in an Apache-2.0 repository. The efficient classical training objective does not give efficient exact sampling of every trained circuit, and the paper's scaling results do not establish an advantage for our small examples. [Research paper](https://arxiv.org/abs/2503.02934) [Experiment code](https://github.com/XanaduAI/scaling-gqml)

The accompanying IQPopt package uses JAX to estimate expectation values and train models. Its generators specify products of Pauli-X operators. A Qiskit implementation can translate those generators into equivalent parameterized rotations, then verify the exported circuit against an independent small-state simulation. Preserve parameter conventions and bit ordering in these checks. This adapter is proposed work; the local preflight below did not install or validate IQPopt. [IQPopt paper](https://arxiv.org/abs/2501.04776) [Package and generator conventions](https://github.com/XanaduAI/iqpopt)

The proposed extension asks: does restricting learned interactions to fit a target device improve sample quality after noisy execution, compared with training a dense model and routing it afterward? Start at 6-10 qubits, compare dense, hardware-local and equally sparse random interactions, and freeze the loss and optimization budget. Check both ideal and noisy distributions so reduced expressiveness is visible alongside reduced circuit cost. Report native two-qubit gate counts, depth, sample quality and classical training cost. Give the sparse controls comparable parameter counts and report any remaining differences.

This is a concrete implementation and benchmarking contribution. A novelty claim would need a narrower literature review. Adaptive-kernel training already has follow-up work, so it should be treated as a reproduction option rather than a new idea. [Kernel-adaptive IQP study](https://arxiv.org/abs/2510.08476)

## 2. QCNN: classifying phases of quantum states

Cong, Choi and Lukin introduced quantum convolutional neural networks for recognizing a one-dimensional symmetry-protected topological phase and for learning error-correction procedures. The model alternates local processing with pooling that reduces the active system. [Original paper, Nature Physics 15, 1273-1278 (2019)](https://arxiv.org/abs/1810.03787)

Qiskit Machine Learning provides a maintained QCNN tutorial with an eight-to-four-to-two-to-one circuit. Its example classifies horizontal and vertical image lines; rerunning that notebook would not reproduce the original quantum-phase experiment. Its circuit construction and training code are useful starting points under the repository's Apache-2.0 license. No original-author repository for the 2019 paper was verified in this search. [Official tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html) [Qiskit Machine Learning license](https://github.com/qiskit-community/qiskit-machine-learning/blob/main/LICENSE.txt)

### Data and proposed extension

Qiskit Machine Learning 0.9.1 includes `phase_of_matter_data`. It generates ground states of four small spin-chain Hamiltonians, including a bond-alternating Heisenberg model with trivial and topological labels. Its default path samples parameters inside named phase regions and uses sparse exact diagonalization. It returns amplitudes or `Statevector` objects. This is accessible simulated quantum data, rather than a downloaded experimental measurement table. [Dataset API](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.datasets.phase_of_matter_data.html) [0.9.1 release](https://github.com/qiskit-community/qiskit-machine-learning/releases/tag/0.9.1)

Start with the binary Heisenberg task at four and eight qubits. Hold out entire coupling-parameter intervals, not merely random adjacent points. Record the generator's phase conventions and treat the boundary region separately because finite systems do not have the same sharp phase transition as the infinite model.

The extension is a measurement-budget study: compare the trained QCNN with logistic regression and an RBF SVM trained on measured local Pauli expectations and correlations. Add a classical-shadow baseline if the first comparison works. Give each method the same total number of state preparations for training and a separately matched inference budget. Training the QCNN repeatedly consumes new state preparations; measuring one classical feature table once also has a cost. Report both.

This comparison matters because Bermejo et al. show that common QCNN benchmark tasks can be learned through classical simulation assisted by Pauli shadows. Their findings make a useful testable question for the project: how do the approaches compare when shots, noise and preparation costs are included? [PRX Quantum 7, 020304 (2026)](https://arxiv.org/abs/2408.12739)

Report balanced accuracy, calibration or Brier score, accuracy against distance from the phase boundary, robustness to noise, and total state preparations. An exact-amplitude classifier can be a simulator diagnostic, but should have its own label because it receives more information than a measurement-based learner. Do not pass Hamiltonian parameters or their phase labels into a classifier as features.

### Preparation and hardware limits

`phase_of_matter_data` does not return an efficient ground-state-preparation circuit. Its source also says that a non-`None` backend currently invokes a VQE path using `StatevectorEstimator` regardless of the backend object; this is not a functioning hardware-data acquisition path. Use `backend=None` for the reference dataset. [Generator source](https://qiskit-community.github.io/qiskit-machine-learning/_modules/qiskit_machine_learning/datasets/phase_of_matter/phase_of_matter.html)

Qiskit's `StatePreparation` can synthesize a circuit from amplitudes through isometry synthesis. That convenience does not make arbitrary amplitude loading cheap. Transpile and count the preparation gates separately from the QCNN. The hardware version should either accept that measured preparation cost or use a small family with an explicitly shallow preparation circuit. [StatePreparation API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.StatePreparation)

This is feasible as a local simulation project. A hardware demonstration should begin at four qubits, with frozen learned parameters and a documented preparation circuit. Eight-qubit exact-state initialization may cost more than the classifier itself.

## 3. Quantum autoencoder: compression and fault detection

Romero, Olson and Aspuru-Guzik introduced a trainable quantum autoencoder that compresses a family of quantum states into fewer qubits. Their demonstrations include molecular and Hubbard-model states. Qiskit's official tutorial provides an implementation and a five-input-qubit example compressed to three latent qubits. The tutorial's SWAP-test circuit uses eight total qubits because it adds reference and auxiliary registers. [Original paper, Quantum Science and Technology 2, 045001 (2017)](https://arxiv.org/abs/1612.02806) [Official Qiskit tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/12_quantum_autoencoder.html)

The official notebook is available in the Apache-2.0 Qiskit Machine Learning repository. No original-author code for the 2017 paper was verified here. QAE anomaly detection itself is established, including a high-energy physics study on classically encoded event data; that application does not provide live coherent quantum inputs. [Notebook source](https://github.com/qiskit-community/qiskit-machine-learning/blob/main/docs/tutorials/12_quantum_autoencoder.ipynb) [Ngairangbam et al., Physical Review D 105, 095004 (2022)](https://arxiv.org/abs/2112.04958)

### Proposed reproduction and extension

Reproduce compression on a four-to-six-qubit state family before introducing anomaly labels. Train on a range of normal state-preparation parameters and test compression on a held-out range. A small spin-chain family gives a physics-based simulation task. A deliberately compressible shallow circuit family gives a hardware-friendly control with a known solution; label it as a constructed control.

Then introduce held-out faults: coherent gate miscalibration, local bit or phase errors, and parameter changes that stay within the learned family. Score an input using the probability that its discarded register differs from the zero reference. When the reference is a known zero state, measuring that register directly estimates the score and avoids a SWAP-test auxiliary circuit. Keep the SWAP-test implementation as a small-case agreement check if needed.

The question is whether a compression-based detector notices the faults we care about. A fault confined to the learned latent subspace can preserve the compression score; include such cases instead of designing only easily detected anomalies. Compare with fixed stabilizer or physical-witness measurements where applicable, and classical detectors on measured local features with the same state-copy budget. Add an exact-density-matrix reference only as a simulation diagnostic.

Report reconstruction fidelity after discarding and resetting the trash register, compression ratio, AUROC, AUPRC at a declared anomaly prevalence, false alarms on held-out normal states, and detection by fault type. Choose thresholds using normal validation data, and show robustness when the noise differs between training and deployment. This supports a portfolio claim about building and evaluating a quantum state-compression pipeline, even if a simpler witness is better at detecting a particular fault.

## 4. ML-guided Qiskit compilation

Compilation translates an abstract circuit into operations allowed by a device. Different layouts and pass choices can change gate count, depth and estimated error. MQT Predictor combines supervised learning for device selection with reinforcement learning for compilation choices. It is an existing research project with public code, and MQT Bench supplies circuit families as Qiskit circuits. This is classical ML applied to quantum computing, which satisfies the Qiskit route in the portfolio brief. [MQT Predictor paper, ACM TQC (2025)](https://arxiv.org/abs/2310.06889) [Predictor code](https://github.com/munich-quantum-toolkit/predictor) [MQT Bench, MIT license](https://github.com/munich-quantum-toolkit/bench)

A manageable first version would learn to choose among a small, frozen set of Qiskit compilation configurations. Start with a few hundred small circuits and a fixed local target. Compare the selector with a fixed preset, random configuration selection under the same search budget, and the best configuration found by trying the complete candidate set. Count data-generation and training costs separately from deployment cost. This is an extension inspired by MQT Predictor, not a reproduction of its full RL framework.

The research increment is generalization under a limited compilation budget: hold out entire algorithm families and evaluate on changed error maps or connectivity. Use extra search only when the selector is uncertain, and charge those additional compilations to its budget. Measure selection regret, compilation latency, two-qubit gates and depth. Treat an error-product estimate as a proxy, and validate a small subset through noisy simulation. Verify circuit equivalence with output layouts accounted for on tractable instances.

This can produce a useful command-line tool: input a circuit and a target, receive a compiled circuit plus a cost comparison. It needs no quantum hardware for its main experiments. The complete current MQT Predictor requires training its own models; its documentation does not promise ready-made models for every configuration. [Framework setup](https://mqt.readthedocs.io/projects/predictor/en/stable/setup.html)

## 5. Warm-start QAOA

Egger, Marecek and Woerner initialize quantum optimization using a classical relaxation. Their experiments investigate benefits at shallow circuit depth. IBM now provides a runnable Qiskit tutorial comparing standard and warm-start QAOA on Max-Cut. [Original paper, Quantum 5, 479 (2021)](https://doi.org/10.22331/q-2021-06-17-479) [Current Qiskit tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/warm-start-qaoa)

Reproduce a small case, then test parameter reuse across related weighted Max-Cut instances. For example, alter a small fraction of weights and compare a fresh optimization with initialization from a previously solved instance. This is a proposed extension for the portfolio, not a claim that parameter transfer is new. It connects the algorithm to repeated allocation problems while keeping exact optima available for the first 6-12-variable benchmark.

Compare standard QAOA, warm-start QAOA, a strong classical local-search or annealing method, and the classical relaxation with rounding alone. Record exact optimum gaps, optimum-hit probability, total objective calls, shots and wall time. Charge relaxation cost, parameter-learning cost, tuning and any reused training instances. Match native circuit costs and test whether a benefit survives finite sampling and noise. A visual partitioning demo would make the result easy to inspect.

Use the current SDK directly or `qiskit-addon-opt-mapper`. The older `qiskit-optimization` repository was archived on 12 July 2026 and directs users to the newer mapper. [Archived project notice](https://github.com/qiskit-community/qiskit-optimization) [Current mapper, Apache-2.0](https://github.com/Qiskit/qiskit-addon-opt-mapper)

## Chemistry option for later

Sample-based quantum diagonalization (SQD) would add quantum chemistry and hybrid numerical methods to the portfolio. The research and Qiskit addon are public, and a small experiment could compare recovery methods and sample budgets against an exact energy. The large published calculations use substantial classical computing resources. The official chemistry tutorial also depends on PySCF, which requires WSL on this Windows machine. It is a reasonable later project after a smaller Qiskit release. [Research paper](https://arxiv.org/abs/2405.05068) [SQD addon, Apache-2.0](https://github.com/Qiskit/qiskit-addon-sqd) [Tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization) [PySCF installation](https://pyscf.org/user/install.html)

## Shared implementation decisions

Use current V2 primitives. Old examples using `QuantumInstance`, `CircuitQNN`, `OpflowQNN`, or V1 primitives need migration. Current tutorials use the function forms such as `real_amplitudes`; API versions should be pinned in a fresh environment for the new project. [Qiskit Machine Learning release notes](https://qiskit-community.github.io/qiskit-machine-learning/release_notes.html)

Qiskit Aer supports local noise models. Exact statevector results, shot-only results, noisy simulation and real-device results should remain distinct in reports. Aer options are not identical to IBM Runtime options; a simulated noise model is not hardware validation. [IBM guide to exact and noisy simulation](https://quantum.cloud.ibm.com/docs/en/guides/simulate-with-qiskit-aer)

For scale, a single dense complex128 statevector at twelve qubits occupies 64 KiB; the corresponding dense density matrix occupies 256 MiB. Those are array sizes, not total process memory or runtime estimates. Repeated circuit evaluations, noise trajectories, training examples and extra registers can dominate the cost. Four-to-eight-qubit experiments leave more room for useful repetitions and controls.

A quantum state available before measurement, a classical amplitude vector produced by simulation, and a table of previously measured outcomes support different experiments. A downloaded table cannot be treated as if coherent copies of its source state are available. State-preparation assumptions belong in every experiment configuration.

### Local feasibility check

On 23 October 2026, an isolated `uv run --no-project` environment on native Windows successfully ran Python 3.11.15 with Qiskit 2.6.1, Qiskit Aer 0.18.0 and Qiskit Machine Learning 0.9.1. The check sampled a Bell circuit with `StatevectorSampler`, ran the same circuit through Aer with a two-qubit depolarizing error, and generated normalized four-qubit phase-classification states. Each sampling run returned the requested 1,024 measurements. This confirms those basic APIs work locally; no candidate research result has been reproduced yet. The neuroimaging environment and lockfile were not changed.

## First deliverable and stopping point

For the QCBM project, create a separate repository rather than mixing the implementation into the neuroimaging experiment history. An initial release can be bounded to:

1. Six-qubit reproduction on Gaussian mixtures and 2-by-3 Bars-and-Stripes, with three fixed seeds and exact-state reference results.
2. One frozen circuit architecture, two MMD losses, 256 and 1,024 shots per objective evaluation, three seeds, and a cap of 200 objective evaluations per fit. Count every optimizer evaluation, including gradient or SPSA perturbations. Add a small noise sweep only after these runs work.
3. Classical baselines, saved configurations and result tables, automated checks for sampling and losses, a short demo that shows generated samples, and a report explaining failures as well as successes.

The acceptance criterion is a reproducible comparison with an explicit resource ledger. After that release, the IQP/Qiskit connectivity study is the main research increment. ML-guided compilation and warm-start QAOA are separate portfolio projects that add different skills. Hardware access is optional for the first release and should be a separately budgeted experiment.
