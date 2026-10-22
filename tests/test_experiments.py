import json
from pathlib import Path

import numpy as np
import pytest

from quantum_neuro_ml.evaluation import EvaluationData, evaluate_models, sample_training_indices
from quantum_neuro_ml.experiments import bottleneck_plans, paired_comparisons
from quantum_neuro_ml.experiments import validate_configuration
from quantum_neuro_ml.data.cohort import CohortPolicy


def test_training_subsamples_use_no_test_subjects_and_are_shared_reproducibly():
    ids = np.arange(48)
    labels = ids % 2
    groups = np.repeat(list("ABCD"), 12)
    data = EvaluationData(np.zeros((48, 2)), labels, groups, ids)
    train = np.flatnonzero(groups != "D")
    quarter = sample_training_indices(data, train, 0.25, 42)
    half = sample_training_indices(data, train, 0.5, 42)
    np.testing.assert_array_equal(quarter, sample_training_indices(data, train, 0.25, 42))
    assert set(quarter) <= set(half) <= set(train)
    assert "D" not in groups[quarter]
    for group in "ABC":
        assert all(np.sum((groups[quarter] == group) & (labels[quarter] == label)) >= 2 for label in (0, 1))


def test_nested_classical_and_quantum_readouts_receive_identical_representations(tmp_path: Path):
    pytest.importorskip("pennylane")
    config = json.loads(Path("config/experiments-v2.json").read_text())
    config.update(pca_rank=2, pca_dimensions=[2], kernel_C=[1], logistic_C=[0.1],
                  gamma_factors=[1], polynomial_degrees=[2], quantum_depths=[1])
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1], 16)
    groups = np.repeat(list("ABCD"), 8)
    values = np.column_stack([rng.uniform(10, 40, 32), rng.choice([1, 2], 32),
                              rng.uniform(0, 0.2, 32), rng.normal(size=(32, 5))])
    data = EvaluationData(values, labels, groups, np.arange(1, 33))
    report, predictions = evaluate_models(data, bottleneck_plans(config, tmp_path / "cache"), verbose=False)
    paired = paired_comparisons(report, 42)
    assert len(paired) == 6
    assert len(predictions) == 32 * 8
    for group in "ABCD":
        for kind in ("raw", "residual"):
            folds = [f for f in report["folds"] if f["held_out_group"] == group and f["model"].startswith(kind)]
            assert len({f["diagnostics"]["training_representation_sha256"] for f in folds}) == 1
            assert len({f["diagnostics"]["test_representation_sha256"] for f in folds}) == 1


@pytest.mark.parametrize("field,value", [("inner_splits", 5), ("quantum_shots", 100),
                                        ("quantum_device", "other"), ("cohort_policy", "other"),
                                        ("angle_mapping", "other")])
def test_configuration_cannot_claim_unimplemented_semantics(field, value):
    config = json.loads(Path("config/experiments-v2.json").read_text())
    config[field] = value
    with pytest.raises(ValueError, match=f"Unsupported {field}"):
        validate_configuration(config, CohortPolicy.load(Path("config/cohort-v1.json")))


def test_held_out_perturbation_cannot_change_pca_scaling_or_nuisance_fits(tmp_path: Path):
    from joblib import Memory
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from quantum_neuro_ml.evaluation import ModelPlan
    from quantum_neuro_ml.representations import EdgeFeatures, NuisanceResidualizer, FirstComponents, BoundedAngles
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1], 16)
    groups = np.repeat(list("ABCD"), 8)
    values = np.column_stack([rng.uniform(10, 40, 32), rng.choice([1, 2], 32),
                              rng.uniform(0, 0.2, 32), rng.normal(size=(32, 5))])
    plans = []
    for name, initial in (("raw", EdgeFeatures()), ("residual", NuisanceResidualizer())):
        pipeline = Pipeline([("input", initial), ("scale", StandardScaler()),
                             ("pca", PCA(n_components=2, random_state=42, svd_solver="randomized")),
                             ("components", FirstComponents(2)), ("scale_components", StandardScaler()),
                             ("angles", BoundedAngles()), ("model", LogisticRegression(solver="liblinear"))],
                            memory=Memory(tmp_path / "cache", verbose=0))
        plans.append(ModelPlan(name, pipeline, {"model__C": [0.1, 1]}))
    first, _ = evaluate_models(EvaluationData(values, labels, groups, np.arange(1, 33)), plans, verbose=False)
    changed = values.copy()
    changed[groups == "A", 0] += 100
    changed[groups == "A", 2] += 0.1
    changed[groups == "A", 3:] += 1000
    second, _ = evaluate_models(EvaluationData(changed, labels, groups, np.arange(1, 33)), plans, verbose=False)
    for name in ("raw", "residual"):
        a = next(f for f in first["folds"] if f["model"] == name and f["held_out_group"] == "A")
        b = next(f for f in second["folds"] if f["model"] == name and f["held_out_group"] == "A")
        assert a["selected_parameters"] == b["selected_parameters"]
        assert a["inner_balanced_accuracy"] == b["inner_balanced_accuracy"]
        assert a["diagnostics"]["training_representation_sha256"] == b["diagnostics"]["training_representation_sha256"]
        assert a["diagnostics"]["pca_explained_variance_ratio"] == b["diagnostics"]["pca_explained_variance_ratio"]
