import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantum_neuro_ml.graph_encoder import SignedGraphEncoder, edge_matrices, graph_structure, make_network


def test_signed_structure_and_spectral_coordinates():
    rng = np.random.default_rng(4)
    edges = rng.normal(size=(3, 15))
    matrix = edge_matrices(edges)
    p, n, spectral, gap = graph_structure(edges, 2)
    degree = 1 + np.abs(matrix).sum(axis=-1)
    expected = matrix / np.sqrt(degree[:, :, None] * degree[:, None, :])
    np.testing.assert_allclose(p - n, expected, atol=1e-7)
    np.testing.assert_allclose(p, p.transpose(0, 2, 1))
    assert np.all(p >= 0) and np.all(n >= 0)
    assert spectral.shape == (3, 6, 2) and gap >= 0
    np.testing.assert_allclose(spectral.transpose(0, 2, 1) @ spectral,
                               np.broadcast_to(np.eye(2), (3, 2, 2)), atol=1e-6)
    for coords in spectral:
        assert (coords[np.argmax(abs(coords), axis=0), np.arange(2)] > 0).all()
    with pytest.raises(ValueError, match="Edge count"):
        edge_matrices(np.ones((2, 8)))


def test_graph_messages_are_invariant_to_node_order():
    generator = torch.Generator().manual_seed(23)
    features = torch.randn(2, 6, 8, generator=generator)
    positive = torch.rand(2, 6, 6, generator=generator)
    negative = torch.rand(2, 6, 6, generator=generator)
    network = make_network(8, 4, 2, 0.2).eval()
    order = torch.tensor([3, 1, 4, 0, 5, 2])
    with torch.no_grad():
        a = network.encode(features, positive, negative)
        b = network.encode(features[:, order], positive[:, order][:, :, order], negative[:, order][:, :, order])
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_graph_fit_scaling_repeatability_and_cache_round_trip(tmp_path):
    import joblib

    rng = np.random.default_rng(8)
    X = rng.normal(size=(24, 15))
    y = np.tile([0, 1], 12)
    X[:, :4] += y[:, None] * 2
    settings = dict(spectral_dimensions=2, hidden_units=4, embedding_dimensions=2, epochs=12,
                    learning_rate=0.01, batch_size=8, dropout=0.2)
    state = torch.random.get_rng_state().clone()
    a = SignedGraphEncoder(**settings).fit(X, y)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    b = SignedGraphEncoder(**settings).fit(X, y)
    np.testing.assert_allclose(a.transform(X), b.transform(X), rtol=0, atol=0)
    np.testing.assert_allclose(a.scaler_.mean_, X.mean(axis=0))
    assert a.diagnostics_["final_training_loss"] < a.diagnostics_["initial_training_loss"]
    before = a.scaler_.mean_.copy()
    a.transform(X[:2] + 100)
    np.testing.assert_array_equal(a.scaler_.mean_, before)
    joblib.dump(a, tmp_path / "graph.joblib")
    restored = joblib.load(tmp_path / "graph.joblib")
    np.testing.assert_array_equal(a.transform(X), restored.transform(X))


def test_graph_nested_shared_readouts(tmp_path):
    pytest.importorskip("pennylane")
    from quantum_neuro_ml.evaluation import EvaluationData, evaluate_models
    from quantum_neuro_ml.experiments import paired_comparisons
    from quantum_neuro_ml.graph_experiment import graph_plans

    rng = np.random.default_rng(3)
    n = 32
    X = np.column_stack([rng.uniform(10, 30, n), np.tile([1, 2, 2, 1], 8),
                         rng.uniform(0, 0.15, n), rng.normal(size=(n, 15))])
    y, groups = np.tile([0, 1], 16), np.repeat(["A", "B", "C", "D"], 8)
    config = json.loads(Path("config/graph-v4.json").read_text())
    config.update(spectral_dimensions=2, hidden_units=3, embedding_dimensions=2, epochs=2, batch_size=8)
    kernels = json.loads(Path("config/experiments-v2.json").read_text())
    kernels.update(logistic_C=[0.1], kernel_C=[1.0], gamma_factors=[1.0],
                   polynomial_degrees=[2], quantum_depths=[1])
    plans = graph_plans(config, kernels, tmp_path / "cache")
    report, rows = evaluate_models(EvaluationData(X, y, groups, np.arange(n)), plans, verbose=False)
    assert len(rows) == n * 8
    assert len(paired_comparisons(report, 42)) == 6
    for fold in report["folds"]:
        assert fold["diagnostics"]["graph"]["training_subjects"] == 24


def test_graph_cache_snapshot_copies_exact_bytes(tmp_path):
    import hashlib
    import json

    from quantum_neuro_ml.graph_experiment import snapshot_cache

    source = tmp_path / "source-cache"
    source.mkdir()
    payload = source / "nested" / "transform.pkl"
    payload.parent.mkdir()
    payload.write_bytes(b"cached representation")
    destination = tmp_path / "copied-cache"
    manifest_path = tmp_path / "cache-manifest.json"
    manifest = snapshot_cache(source, destination, manifest_path)
    assert manifest == {"nested/transform.pkl": hashlib.sha256(b"cached representation").hexdigest()}
    assert json.loads(manifest_path.read_text()) == manifest
    assert (destination / "nested" / "transform.pkl").read_bytes() == b"cached representation"
