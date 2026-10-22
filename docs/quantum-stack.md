# Quantum software stack

Decision date: 2026-10-22. The exact-simulator kernel implementation is installed and its first cohort comparisons are running. The variational implementation must pass its separate gradient and trainability gates before cohort fitting.

Use PennyLane for quantum feature maps and variational readouts, starting with its local `default.qubit` simulator. PennyLane documents [backpropagation support and execution tracking](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.DefaultQubit.html) and a [PyTorch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html) through which circuit outputs participate in automatic differentiation. This supports a PyTorch graph encoder and keeps the kernel and variational experiments in one framework. The locked environment contains PennyLane 0.45.1 and PyTorch 2.14.1.

The first kernel experiment uses 4-, 6-, or 8-dimensional representations fitted within training folds, an explicitly specified angle-encoding circuit, and fidelity kernels. The implementation computes squared state overlaps, then passes training and held-out-to-training matrices to scikit-learn's precomputed-kernel SVM. Compare with classical kernels on exactly the same compressed representation and folds. The circuit and search space were frozen in the [v2 protocol](follow-up-protocol.md) before its runs.

For a later variational readout, use the PyTorch interface and analytic simulator gradients first. Treat finite-shot evaluation and its compatible gradient method as separate sensitivity experiments. Record precision, qubits, gates, depth, shots, seeds, and execution counts as well as runtime.

Qiskit Machine Learning also supports [fidelity-kernel experiments](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html). Reconsider it if the optional IBM hardware experiment needs a native integration. The initial simulator study does not require cloud credentials.

The optional `quantum` and `graph` dependency groups are locked in `uv.lock`. Kernel preflight tests validate a known one-qubit fidelity, full circuit states against an independent NumPy implementation, kernel symmetry and positive semidefiniteness, and sklearn fitting. Variational readouts additionally require numerical gradient and parameter-activity checks. The classical environment remains the reference for cohort and split behaviour.
