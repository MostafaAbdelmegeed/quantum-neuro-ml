import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pennylane")

spec = importlib.util.spec_from_file_location("graph_variational", Path("scripts/graph_variational.py"))
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_frozen_graph_readouts_reproduce_reference_coordinates(tmp_path):
    from quantum_neuro_ml.evaluation import EvaluationData, evaluate_models
    from quantum_neuro_ml.graph_experiment import graph_plans
    from quantum_neuro_ml.provenance import recorded_run
    from quantum_neuro_ml.variational_experiment import paired_difference

    rng = np.random.default_rng(31)
    n = 32
    X = np.column_stack([rng.uniform(10, 30, n), np.tile([1, 2, 2, 1], 8),
                         rng.uniform(0, 0.15, n), rng.normal(size=(n, 15))])
    data = EvaluationData(X, np.tile([0, 1], 16), np.repeat(["A", "B", "C", "D"], 8), np.arange(n))
    graph = json.loads(Path("config/graph-v4.json").read_text())
    graph.update(spectral_dimensions=2, hidden_units=3, epochs=2, batch_size=8)
    kernels = json.loads(Path("config/experiments-v2.json").read_text())
    kernels.update(logistic_C=[0.1])
    readout = json.loads(Path("config/variational-v3.json").read_text())
    readout.update(seeds=[42], epochs=2, batch_size=8, learning_rates=[0.01])
    reference_dir = tmp_path / "reference"
    templates = [p for p in graph_plans(graph, kernels, reference_dir / "cache") if p.name.endswith("_logistic")]
    reference, _ = evaluate_models(data, templates, verbose=False)
    reference.update(dataset_sha256="synthetic", protocol_configuration=graph, kernel_configuration=kernels)
    with recorded_run(reference_dir, {"experiment": "synthetic-graph-reference"}, {}) as receipt:
        receipt.finish(reference)
    restored = experiment.validate_reference(reference_dir / "report.json", "synthetic", graph, kernels)
    plans = experiment.trainable_plans(graph, kernels, readout, reference_dir / "cache")
    report, rows = evaluate_models(data, plans, verbose=False,
                                   on_fold=lambda fold, predictions: experiment.align_reference(fold, restored))
    assert len(rows) == n * 4
    for kind in ("raw", "residual"):
        comparison = paired_difference(report, f"{kind}_graph4_seed42_quantum", f"{kind}_graph4_seed42_mlp")
        assert len(comparison["balanced_accuracy"]["by_group"]) == 4
    bad = deepcopy(report["folds"][0])
    bad["diagnostics"]["training_representation_sha256"] = "changed"
    with pytest.raises(ValueError, match="Frozen graph coordinates"):
        experiment.align_reference(bad, restored)
    with pytest.raises(ValueError, match="dataset or configuration"):
        experiment.validate_reference(reference_dir / "report.json", "different", graph, kernels)


def test_combination_configuration_cannot_change_encoder_seed():
    config = json.loads(Path("config/graph-variational-v6.json").read_text())
    graph = json.loads(Path("config/graph-v4.json").read_text())
    readout = json.loads(Path("config/variational-v3.json").read_text())
    experiment.validate_combination(config, graph, readout)
    graph["seed"] = 137
    with pytest.raises(ValueError, match="do not match"):
        experiment.validate_combination(config, graph, readout)
