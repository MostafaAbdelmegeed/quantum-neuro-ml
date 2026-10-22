"""Small, parameter-matched trainable readouts for bounded coordinates."""

from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def quantum_expectations(angles, weights):
    """Return a differentiable batch of Pauli-Z expectations."""
    import torch

    dimensions = angles.shape[-1]
    circuit = _circuit(dimensions, weights.shape[0])
    return torch.stack(circuit(angles, weights), dim=-1)


def _circuit(dimensions: int, depth: int):
    # A QNode is cheap compared with training, but reusing it avoids rebuilding
    # the device at each minibatch. Each estimator owns its execution counters.
    import pennylane as qml

    @qml.qnode(qml.device("default.qubit", wires=dimensions, shots=None), interface="torch", diff_method="backprop")
    def circuit(angles, weights):
        for wire in range(dimensions):
            qml.RY(angles[..., wire], wires=wire)
        for layer in range(depth):
            for wire in range(dimensions):
                qml.RZ(weights[layer, wire, 0], wires=wire)
                qml.RY(weights[layer, wire, 1], wires=wire)
            for wire in range(dimensions):
                qml.CNOT(wires=[wire, (wire + 1) % dimensions])
        return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(dimensions))

    return circuit


class TrainableReadout(ClassifierMixin, BaseEstimator):
    """Fit a fixed training schedule; expose the usual sklearn classifier seam."""

    def __init__(self, family="quantum", learning_rate=0.01, epochs=60, batch_size=32,
                 weight_decay=0.0001, seed=42, depth=2, hidden_units=3):
        self.family = family
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.depth = depth
        self.hidden_units = hidden_units

    def fit(self, X, y):
        import torch
        from torch import nn

        X, y = check_X_y(X, y, dtype=np.float64)
        if set(np.unique(y)) != {0, 1} or self.family not in ("quantum", "mlp"):
            raise ValueError("Expected a binary target and quantum or mlp family")
        if not 2 <= X.shape[1] <= 8 or np.max(np.abs(X)) > np.pi + 1e-12:
            raise ValueError("Readouts require two to eight bounded angle coordinates")
        if (self.depth not in (1, 2) or self.hidden_units < 1 or self.epochs < 1 or self.batch_size < 1
                or not np.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not np.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError("Invalid readout training settings")
        started = time.perf_counter()
        torch.set_num_threads(1)
        dimensions = X.shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            if self.family == "mlp":
                self.model_ = nn.Sequential(nn.Linear(dimensions, self.hidden_units), nn.Tanh(),
                                            nn.Linear(self.hidden_units, 1)).double()
                self.weights_ = None
            else:
                self.weights_ = nn.Parameter(torch.randn(self.depth, dimensions, 2, dtype=torch.float64) * 0.1)
                self.model_ = nn.Linear(dimensions, 1).double()
                self.circuit_ = _circuit(dimensions, self.depth)
        parameters = list(self.model_.parameters())
        if self.weights_ is not None:
            parameters.append(self.weights_)
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        features = torch.as_tensor(X, dtype=torch.float64)
        labels = torch.as_tensor(y, dtype=torch.float64)
        counts = np.bincount(y.astype(int), minlength=2)
        sample_weights = torch.as_tensor(len(y) / (2 * counts[y.astype(int)]), dtype=torch.float64)
        self.forward_calls_, self.examples_processed_ = 0, 0

        def loss(indices):
            logits = self._logits(features[indices])
            values = nn.functional.binary_cross_entropy_with_logits(logits, labels[indices], reduction="none")
            return (values * sample_weights[indices]).mean()

        with torch.no_grad():
            initial = float(loss(torch.arange(len(y))))
        generator = torch.Generator().manual_seed(self.seed)
        updates = 0
        for _ in range(self.epochs):
            order = torch.randperm(len(y), generator=generator)
            for indices in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                value = loss(indices)
                if not torch.isfinite(value):
                    raise ValueError("Readout training produced a nonfinite loss")
                value.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise ValueError("Readout gradient is missing or nonfinite")
                optimizer.step()
                updates += 1
        with torch.no_grad():
            final = float(loss(torch.arange(len(y))))
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = dimensions
        self.diagnostics_ = {
            "family": self.family, "seed": self.seed, "parameters": sum(p.numel() for p in parameters),
            "epochs": self.epochs, "optimizer": "Adam", "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay, "batch_size": self.batch_size,
            "updates": updates, "initial_training_loss": initial, "final_training_loss": final,
            "fit_seconds": time.perf_counter() - started, "training_subjects": len(y),
            "dtype": "float64", "hardware": "CPU", "torch_version": torch.__version__,
            "training_forward_calls": self.forward_calls_, "training_examples_processed": self.examples_processed_,
        }
        if self.family == "quantum":
            import pennylane as qml
            self.diagnostics_.update(qubits=dimensions, depth=self.depth, shots=None,
                                     gradient="backprop", device="default.qubit", pennylane_version=qml.__version__,
                                     gates_per_example={"RY": dimensions * (self.depth + 1),
                                                        "RZ": dimensions * self.depth, "CNOT": dimensions * self.depth})
        return self

    def _logits(self, features):
        import torch

        self.forward_calls_ += 1
        self.examples_processed_ += len(features)
        if self.weights_ is not None:
            features = torch.stack(self.circuit_(features, self.weights_), dim=-1)
        return self.model_(features).squeeze(-1)

    def decision_function(self, X):
        import torch

        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_ or np.max(np.abs(X)) > np.pi + 1e-12:
            raise ValueError("Prediction coordinates differ from the fitted bounded input")
        with torch.no_grad():
            result = np.concatenate([self._logits(batch).numpy() for batch in
                                     torch.as_tensor(X, dtype=torch.float64).split(self.batch_size)])
        return result

    def predict_proba(self, X):
        score = self.decision_function(X)
        probability = np.exp(-np.logaddexp(0, -score))
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
