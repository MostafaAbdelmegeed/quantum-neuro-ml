import numpy as np
import pytest

pytest.importorskip("pennylane")
from quantum_neuro_ml.quantum_kernels import feature_states, fidelity_matrix, FidelityKernelSVC


def independent_state(angles, depth):
    wires = len(angles)
    state = np.zeros(2 ** wires, dtype=complex)
    state[0] = 1
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    def single_gate(vector, gate, wire):
        tensor = vector.reshape([2] * wires)
        moved = np.moveaxis(tensor, wire, 0)
        transformed = np.tensordot(gate, moved, axes=([1], [0]))
        return np.moveaxis(transformed, 0, wire).reshape(-1)
    for _ in range(depth):
        for wire in range(wires):
            state = single_gate(state, hadamard, wire)
        for wire, angle in enumerate(angles):
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            state = single_gate(state, np.array([[c, -s], [s, c]]), wire)
        indices = np.arange(len(state))
        for wire in range(wires - 1):
            first = 1 - 2 * ((indices >> (wires - wire - 1)) & 1)
            second = 1 - 2 * ((indices >> (wires - wire - 2)) & 1)
            state *= np.exp(-0.5j * angles[wire] * angles[wire + 1] / np.pi * first * second)
    return state


@pytest.mark.parametrize("wires,depth", [(1, 1), (2, 1), (4, 1), (4, 2)])
def test_simulator_states_match_independent_numpy_circuit(wires, depth):
    coordinates = np.random.default_rng(42).uniform(-np.pi, np.pi, (3, wires))
    actual = feature_states(coordinates, depth)
    expected = np.stack([independent_state(x, depth) for x in coordinates])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    gram = fidelity_matrix(actual, actual)
    np.testing.assert_allclose(np.diag(gram), 1, atol=1e-12)
    assert np.linalg.eigvalsh(gram).min() >= -1e-12


def test_single_qubit_fidelity_matches_closed_form():
    first = np.array([[0.1], [0.8]])
    second = np.array([[-0.4], [1.2], [2.0]])
    actual = fidelity_matrix(feature_states(first), feature_states(second))
    np.testing.assert_allclose(actual, np.cos((first - second.T) / 2) ** 2, atol=1e-12)


def test_quantum_classifier_predicts_and_rejects_changed_dimension():
    X = np.array([[-1, -1], [-0.7, -0.8], [0.7, 0.8], [1, 1]])
    model = FidelityKernelSVC(C=10).fit(X, [0, 0, 1, 1])
    np.testing.assert_array_equal(model.predict(X), [0, 0, 1, 1])
    assert model.diagnostics_["qubits"] == 2
    with pytest.raises(ValueError, match="dimension changed"):
        model.predict(np.zeros((2, 3)))
