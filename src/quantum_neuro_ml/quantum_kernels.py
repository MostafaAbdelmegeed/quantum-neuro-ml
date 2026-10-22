"""Exact, locally simulated fidelity kernels with a scikit-learn classifier."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


@lru_cache(maxsize=32)
def _circuit(wires: int, depth: int):
    device = qml.device("default.qubit", wires=wires, shots=None)

    @qml.qnode(device, interface="numpy")
    def state(angles):
        for _ in range(depth):
            for wire in range(wires):
                qml.Hadamard(wires=wire)
            for wire in range(wires):
                qml.RY(angles[wire], wires=wire)
            for wire in range(wires - 1):
                qml.IsingZZ(angles[wire] * angles[wire + 1] / np.pi, wires=[wire, wire + 1])
        return qml.state()
    return state


@lru_cache(maxsize=50_000)
def _cached_state(wires: int, depth: int, angles: tuple[float, ...]) -> np.ndarray:
    state = np.asarray(_circuit(wires, depth)(np.array(angles)), dtype=np.complex128)
    if not np.isfinite(state).all() or not np.isclose(np.vdot(state, state).real, 1, atol=1e-10, rtol=0):
        raise ValueError("Quantum simulator returned an invalid state")
    state.setflags(write=False)
    return state


def feature_states(X: np.ndarray, depth: int = 1) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or not 1 <= values.shape[1] <= 8 or not np.isfinite(values).all():
        raise ValueError("Expected finite coordinates for 1-8 qubits")
    if depth not in (1, 2) or np.any(np.abs(values) > np.pi + 1e-12):
        raise ValueError("Unsupported depth or coordinates outside [-pi, pi]")
    return np.stack([_cached_state(values.shape[1], depth, tuple(row)) for row in values])


def fidelity_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    values = np.abs(first @ second.conj().T) ** 2
    return np.clip(values.real, 0, 1)


def state_cache_statistics() -> dict[str, int]:
    info = _cached_state.cache_info()
    return {"hits": info.hits, "simulator_executions": info.misses, "cached_states": info.currsize}


class FidelityKernelSVC(ClassifierMixin, BaseEstimator):
    """Angle-feature fidelity SVM; preprocessing belongs in the outer pipeline."""

    def __init__(self, C: float = 1.0, depth: int = 1):
        self.C = C
        self.depth = depth

    def fit(self, X, y):
        values = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = values.shape[1]
        self.training_states_ = feature_states(values, self.depth)
        kernel = fidelity_matrix(self.training_states_, self.training_states_)
        if not np.allclose(kernel, kernel.T, atol=1e-10, rtol=0) or not np.allclose(np.diag(kernel), 1, atol=1e-10, rtol=0):
            raise ValueError("Fidelity training kernel is not symmetric with unit diagonal")
        minimum_eigenvalue = float(np.linalg.eigvalsh(kernel)[0])
        if minimum_eigenvalue < -1e-8:
            raise ValueError("Exact fidelity training kernel is not positive semidefinite")
        self.model_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(kernel, y)
        self.classes_ = self.model_.classes_
        off_diagonal = kernel[~np.eye(len(kernel), dtype=bool)]
        self.diagnostics_ = {
            "qubits": self.n_features_in_, "depth": self.depth, "shots": None, "device": "default.qubit",
            "pennylane_version": qml.__version__, "minimum_training_eigenvalue": minimum_eigenvalue,
            "refit_training_state_vectors": len(values), "training_kernel_entries": int(kernel.size),
            "off_diagonal_kernel_mean": float(off_diagonal.mean()),
            "off_diagonal_kernel_std": float(off_diagonal.std()),
        }
        return self

    def _kernel(self, X):
        check_is_fitted(self, "model_")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Quantum classifier input dimension changed")
        return fidelity_matrix(feature_states(values, self.depth), self.training_states_)

    def predict(self, X):
        return self.model_.predict(self._kernel(X))

    def decision_function(self, X):
        return self.model_.decision_function(self._kernel(X))
