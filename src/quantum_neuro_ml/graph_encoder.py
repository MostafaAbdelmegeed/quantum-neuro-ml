"""A training-only supervised signed-connectivity graph representation."""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def edge_matrices(edges: np.ndarray) -> np.ndarray:
    edges = check_array(edges, dtype=np.float64)
    nodes = int((1 + np.sqrt(1 + 8 * edges.shape[1])) / 2)
    if nodes < 3 or nodes * (nodes - 1) // 2 != edges.shape[1]:
        raise ValueError("Edge count does not describe an undirected graph with at least three nodes")
    matrices = np.zeros((len(edges), nodes, nodes), dtype=np.float64)
    i, j = np.triu_indices(nodes, k=1)
    matrices[:, i, j], matrices[:, j, i] = edges, edges
    return matrices


@lru_cache(maxsize=2048)
def _spectral_coordinates(edge_bytes: bytes, nodes: int, dimensions: int) -> tuple[np.ndarray, float]:
    edges = np.frombuffer(edge_bytes, dtype=np.float64).reshape(1, -1)
    absolute = np.abs(edge_matrices(edges)[0])
    degree = absolute.sum(axis=-1)
    inverse = np.zeros(nodes, dtype=np.float64)
    np.divide(1, np.sqrt(degree), out=inverse, where=degree > 0)
    laplacian = np.eye(nodes) - absolute * inverse[:, None] * inverse[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    coordinates = eigenvectors[:, 1:dimensions + 1].copy()
    pivots = coordinates[np.argmax(np.abs(coordinates), axis=0), np.arange(dimensions)]
    coordinates *= np.where(pivots < 0, -1, 1)
    gap = float(np.diff(eigenvalues[:min(nodes, dimensions + 2)]).min())
    coordinates = coordinates.astype(np.float32)
    coordinates.setflags(write=False)
    return coordinates, gap


def graph_structure(edges: np.ndarray, spectral_dimensions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    edges = check_array(edges, dtype=np.float64)
    weights = edge_matrices(edges)
    nodes = weights.shape[-1]
    if not 1 <= spectral_dimensions < nodes:
        raise ValueError("Spectral dimension must be smaller than the node count")
    absolute = np.abs(weights)
    degree = absolute.sum(axis=-1)
    inverse = 1 / np.sqrt(1 + degree)
    normalization = inverse[:, :, None] * inverse[:, None, :]
    positive = np.maximum(weights, 0) * normalization
    negative = np.maximum(-weights, 0) * normalization
    spectral = np.empty((len(edges), nodes, spectral_dimensions), dtype=np.float64)
    minimum_gap = np.inf
    for index in range(len(edges)):
        coordinates, gap = _spectral_coordinates(np.ascontiguousarray(edges[index]).tobytes(), nodes, spectral_dimensions)
        spectral[index] = coordinates
        minimum_gap = min(minimum_gap, gap)
    return positive.astype(np.float32), negative.astype(np.float32), spectral.astype(np.float32), minimum_gap


def make_network(features: int, hidden: int, embedding: int, dropout: float):
    import torch
    from torch import nn

    class SignedLayer(nn.Module):
        def __init__(self, inputs, outputs):
            super().__init__()
            self.own = nn.Linear(inputs, outputs)
            self.positive = nn.Linear(inputs, outputs, bias=False)
            self.negative = nn.Linear(inputs, outputs, bias=False)

        def forward(self, values, positive, negative):
            return torch.relu(self.own(values) + torch.bmm(positive, self.positive(values))
                              + torch.bmm(negative, self.negative(values)))

    class SignedGraphNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = SignedLayer(features, hidden)
            self.second = SignedLayer(hidden, hidden)
            self.dropout = nn.Dropout(dropout)
            self.embedding = nn.Linear(hidden * 2, embedding)
            self.head = nn.Linear(embedding, 1)

        def encode(self, values, positive, negative):
            values = self.dropout(self.first(values, positive, negative))
            values = self.second(values, positive, negative)
            pooled = torch.cat([values.mean(dim=1), values.amax(dim=1)], dim=-1)
            return torch.tanh(self.embedding(pooled))

        def forward(self, values, positive, negative):
            return self.head(self.encode(values, positive, negative)).squeeze(-1)

    return SignedGraphNetwork()


class SignedGraphEncoder(TransformerMixin, BaseEstimator):
    """Learn graph embeddings using only the supplied training participants."""

    def __init__(self, spectral_dimensions=8, hidden_units=16, embedding_dimensions=4,
                 epochs=40, batch_size=16, learning_rate=0.001, weight_decay=0.0001,
                 dropout=0.2, seed=42):
        self.spectral_dimensions = spectral_dimensions
        self.hidden_units = hidden_units
        self.embedding_dimensions = embedding_dimensions
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed

    def _inputs(self, X):
        import torch

        positive, negative, spectral, gap = graph_structure(X, self.spectral_dimensions)
        profiles = edge_matrices(self.scaler_.transform(X)).astype(np.float32)
        features = np.concatenate([profiles, spectral], axis=-1)
        return torch.from_numpy(features), torch.from_numpy(positive), torch.from_numpy(negative), gap

    def fit(self, X, y):
        import torch
        from torch import nn

        X, y = check_X_y(X, y, dtype=np.float64)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Graph supervision requires both binary diagnoses")
        for value in (self.spectral_dimensions, self.hidden_units, self.embedding_dimensions, self.epochs, self.batch_size):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Graph dimensions and training counts must be positive integers")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid graph optimizer or dropout setting")
        started = time.perf_counter()
        torch.set_num_threads(1)
        self.scaler_ = StandardScaler().fit(X)
        features, positive, negative, gap = self._inputs(X)
        labels = torch.as_tensor(y, dtype=torch.float32)
        counts = np.bincount(y.astype(int), minlength=2)
        sample_weights = torch.as_tensor(len(y) / (2 * counts[y.astype(int)]), dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network_ = make_network(features.shape[-1], self.hidden_units, self.embedding_dimensions, self.dropout)
            optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            def loss(indices):
                logits = self.network_(features[indices], positive[indices], negative[indices])
                values = nn.functional.binary_cross_entropy_with_logits(logits, labels[indices], reduction="none")
                return (values * sample_weights[indices]).mean()
            self.network_.eval()
            with torch.no_grad():
                initial = float(loss(torch.arange(len(y))))
            self.network_.train()
            generator = torch.Generator().manual_seed(self.seed)
            updates = 0
            for _ in range(self.epochs):
                order = torch.randperm(len(y), generator=generator)
                for indices in order.split(self.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    value = loss(indices)
                    if not torch.isfinite(value):
                        raise ValueError("Graph training produced a nonfinite loss")
                    value.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.network_.parameters()):
                        raise ValueError("Graph gradient is missing or nonfinite")
                    optimizer.step()
                    updates += 1
            self.network_.eval()
            with torch.no_grad():
                final = float(loss(torch.arange(len(y))))
        self.n_features_in_ = X.shape[1]
        self.diagnostics_ = {
            "architecture": "two_signed_message_layers_mean_max_pool_tanh_embedding",
            "nodes": features.shape[1], "spectral_dimensions": self.spectral_dimensions,
            "minimum_training_spectral_eigengap": gap, "training_subjects": len(y),
            "parameters_including_discarded_training_head": sum(p.numel() for p in self.network_.parameters()),
            "embedding_dimensions": self.embedding_dimensions, "epochs": self.epochs,
            "seed": self.seed, "updates": updates, "learning_rate": self.learning_rate,
            "initial_training_loss": initial, "final_training_loss": final,
            "fit_seconds": time.perf_counter() - started,
            "torch_version": torch.__version__, "dtype": "float32", "hardware": "CPU",
        }
        return self

    def transform(self, X):
        import torch

        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Graph edge count differs from training")
        features, positive, negative, _ = self._inputs(X)
        self.network_.eval()
        with torch.no_grad():
            result = torch.cat([self.network_.encode(features[i:i + self.batch_size], positive[i:i + self.batch_size],
                                                     negative[i:i + self.batch_size])
                                for i in range(0, len(X), self.batch_size)]).numpy()
        if not np.isfinite(result).all():
            raise ValueError("Graph embeddings are nonfinite")
        return result.astype(np.float64)

    def __getstate__(self):
        # Local torch classes are intentionally private. Store their state_dict
        # so joblib can persist fitted transformers without pickling the classes.
        state = self.__dict__.copy()
        if "network_" in state:
            network = state.pop("network_")
            state["_network_state"] = network.state_dict()
        return state

    def __setstate__(self, state):
        weights = state.pop("_network_state", None)
        self.__dict__.update(state)
        if weights is not None:
            import torch
            nodes = self.diagnostics_["nodes"]
            with torch.random.fork_rng(devices=[]):
                self.network_ = make_network(nodes + self.spectral_dimensions, self.hidden_units,
                                             self.embedding_dimensions, self.dropout)
            self.network_.load_state_dict(weights)
            self.network_.eval()
