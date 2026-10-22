import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from quantum_neuro_ml.variational import TrainableReadout, quantum_expectations


def test_quantum_gradients_match_finite_differences_and_batching():
    rng = np.random.default_rng(7)
    x = torch.tensor(rng.normal(size=(3, 4)), dtype=torch.float64, requires_grad=True)
    weights = torch.tensor(rng.normal(scale=0.1, size=(2, 4, 2)), dtype=torch.float64, requires_grad=True)
    batch = quantum_expectations(x, weights)
    individual = torch.stack([quantum_expectations(row, weights) for row in x])
    torch.testing.assert_close(batch, individual, atol=1e-12, rtol=1e-12)
    objective = (batch * torch.tensor([0.2, -0.3, 0.5, 0.7])).sum()
    gradients = torch.autograd.grad(objective, (x, weights))
    assert torch.all(gradients[1].abs() > 1e-6), "Every declared circuit angle must affect the readout"
    step = 1e-6
    for tensor, gradient, indices in ((x, gradients[0], [(0, 0), (1, 3)]),
                                      (weights, gradients[1], [(0, 1, 0), (1, 2, 1)])):
        for index in indices:
            perturbed = tensor.detach().clone()
            perturbed[index] += step
            a, b = (perturbed, weights.detach()) if tensor is x else (x.detach(), perturbed)
            plus = float((quantum_expectations(a, b) * torch.tensor([0.2, -0.3, 0.5, 0.7])).sum())
            perturbed[index] -= 2 * step
            a, b = (perturbed, weights.detach()) if tensor is x else (x.detach(), perturbed)
            minus = float((quantum_expectations(a, b) * torch.tensor([0.2, -0.3, 0.5, 0.7])).sum())
            assert float(gradient[index]) == pytest.approx((plus - minus) / (2 * step), abs=1e-7)
    assert float(gradients[0].abs().max()) > 0.01
    assert float(gradients[1].abs().max()) > 0.01


@pytest.mark.parametrize("family,parameters", [("quantum", 21), ("mlp", 19)])
def test_readout_training_is_repeatable_and_learns(family, parameters):
    rng = np.random.default_rng(19)
    x = rng.uniform(-2, 2, size=(48, 4))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    settings = dict(family=family, seed=42, epochs=30, learning_rate=0.03, batch_size=24)
    a, b = TrainableReadout(**settings).fit(x, y), TrainableReadout(**settings).fit(x, y)
    np.testing.assert_array_equal(a.decision_function(x), b.decision_function(x))
    assert a.diagnostics_["parameters"] == parameters
    assert a.diagnostics_["final_training_loss"] < a.diagnostics_["initial_training_loss"]
    assert (a.predict(x) == y).mean() > 0.65
    np.testing.assert_allclose(a.predict_proba(x).sum(axis=1), 1)
    with pytest.raises(ValueError, match="coordinates"):
        a.predict(x[:, :3])


def test_variational_config_rejects_unsupported_execution():
    import json
    from pathlib import Path
    from quantum_neuro_ml.variational_experiment import validate_readout_config

    config = json.loads(Path("config/variational-v3.json").read_text())
    validate_readout_config(config)
    config["quantum_shots"] = 100
    with pytest.raises(ValueError, match="quantum_shots"):
        validate_readout_config(config)


def test_variational_nested_readouts_share_inputs(tmp_path):
    import json
    from pathlib import Path
    from quantum_neuro_ml.evaluation import EvaluationData, evaluate_models
    from quantum_neuro_ml.variational_experiment import paired_difference, readout_plans

    rng = np.random.default_rng(90)
    n = 32
    X = np.column_stack([rng.uniform(10, 30, n), np.tile([1, 2, 2, 1], 8),
                         rng.uniform(0, 0.15, n), rng.normal(size=(n, 12))])
    y, groups = np.tile([0, 1], 16), np.repeat(["A", "B", "C", "D"], 8)
    config = json.loads(Path("config/variational-v3.json").read_text())
    config.update(representations=["raw"], seeds=[42], epochs=2, batch_size=5)
    representation = json.loads(Path("config/experiments-v2.json").read_text())
    plans = readout_plans(config, representation, tmp_path / "cache")
    report, rows = evaluate_models(EvaluationData(X, y, groups, np.arange(n)), plans, verbose=False)
    assert len(rows) == n * 2
    comparison = paired_difference(report, "raw_pca4_seed42_quantum", "raw_pca4_seed42_mlp")
    assert len(comparison["balanced_accuracy"]["by_group"]) == 4
