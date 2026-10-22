import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantum_neuro_ml.covariation import EdgeCovariationGroups, compare_modules
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models
from quantum_neuro_ml.representations import EdgeFeatures


def synthetic_edges(seed=9, n=96, groups=4, edges_per_group=8):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, groups))
    values = np.column_stack([
        latent[:, module, None] * rng.choice([-1.0, 1.0], size=(1, edges_per_group))
        + 0.15 * rng.normal(size=(n, edges_per_group))
        for module in range(groups)
    ])
    return values


def test_edge_covariation_groups_learn_training_only_and_ignore_labels():
    values = synthetic_edges()
    labels = np.tile([0, 1], len(values) // 2)
    first = EdgeCovariationGroups(dimensions=4, candidate_edges=32).fit(values[:72], labels[:72])
    flipped = EdgeCovariationGroups(dimensions=4, candidate_edges=32).fit(values[:72], 1 - labels[:72])
    changed_test = values.copy()
    changed_test[72:] = np.random.default_rng(4).normal(500, 100, size=changed_test[72:].shape)
    refit = EdgeCovariationGroups(dimensions=4, candidate_edges=32).fit(changed_test[:72], labels[:72])

    assert first.diagnostics_ == flipped.diagnostics_ == refit.diagnostics_
    assert len(first.modules_) == 4
    assert sorted(sum((list(module) for module in first.modules_), [])) == list(range(32))
    assert first.transform(changed_test[72:]).shape == (24, 4)
    assert np.isfinite(first.transform(changed_test[72:])).all()
    assert first.diagnostics_["labels_used_for_learning"] is False


def test_edge_covariation_groups_reject_unavailable_dimensions():
    with pytest.raises(ValueError, match="Invalid edge matrix"):
        EdgeCovariationGroups(dimensions=8, candidate_edges=4).fit(np.ones((12, 4)))


def test_edge_covariation_pipeline_fits_inside_nested_site_evaluation():
    edges = synthetic_edges(n=48, edges_per_group=8)
    covariates = np.column_stack([np.full(48, 20.0), np.tile([1.0, 2.0], 24),
                                  np.full(48, 0.1)])
    features = np.column_stack([covariates, edges])
    labels = np.tile([0, 1], 24)
    groups = np.repeat(["A", "B", "C", "D"], 12)
    pipeline = Pipeline([
        ("input", EdgeFeatures()),
        ("modules", EdgeCovariationGroups(dimensions=4, candidate_edges=32)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(solver="liblinear", class_weight="balanced")),
    ])
    report, _ = evaluate_models(
        EvaluationData(features, labels, groups, np.arange(48)),
        [ModelPlan("raw_covar4_logistic", pipeline, {"model__C": [1.0]})],
        verbose=False,
    )
    assert len(report["folds"]) == 4
    assert all(fold["diagnostics"]["modules"]["labels_used_for_learning"] is False
               for fold in report["folds"])


def test_signed_module_stability_matches_permuted_groups_and_global_sign():
    first = [{1: 1, 2: -1}, {3: 1, 4: 1}]
    second = [{4: -1, 3: -1}, {2: 1, 1: -1}]
    matched, _ = compare_modules(first, second)
    assert matched == 1.0
