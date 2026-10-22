import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from quantum_neuro_ml.baselines import evaluate_baselines


def dataset():
    ids = np.arange(24)
    labels = ids % 2
    groups = np.repeat(["A", "B", "C", "D"], 6)
    features = np.column_stack([ids, labels + np.random.default_rng(42).normal(0, 0.3, 24), np.sin(ids)])
    return features, labels, groups, ids


def test_all_scaler_fits_use_only_complete_training_groups(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    features, labels, groups, ids = dataset()
    fit_ids = []
    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            fit_ids.append(set(X[:, 0].astype(int)))
            return super().fit(X, y, sample_weight=sample_weight)
    monkeypatch.setattr("quantum_neuro_ml.baselines.StandardScaler", RecordingScaler)
    report, predictions = evaluate_baselines(features, labels, groups, ids, models=("logistic",))
    # Four parameter values x three inner folds + one outer refit, for each site.
    assert len(fit_ids) == 4 * (4 * 3 + 1)
    for fold_index, held_out in enumerate(np.unique(groups)):
        fits = fit_ids[fold_index * 13:(fold_index + 1) * 13]
        for fitted_ids in fits:
            assert not fitted_ids & set(ids[groups == held_out])
            for group in np.unique(groups):
                group_ids = set(ids[groups == group])
                assert fitted_ids & group_ids in (set(), group_ids)
        assert fits[-1] == set(ids[groups != held_out])
    assert len(predictions) == len(ids)
    assert sorted(p["subject_id"] for p in predictions) == ids.tolist()
    assert report["acquisition_groups"] == 4


def test_svm_scores_and_dummy_control_have_expected_metric_contract() -> None:
    features, labels, groups, ids = dataset()
    report, predictions = evaluate_baselines(features, labels, groups, ids, models=("dummy", "rbf_svm"))
    dummy = report["summaries"]["dummy"]["site_mean"]
    assert dummy["balanced_accuracy"] == 0.5
    assert dummy["auroc"] == 0.5
    assert all(p["probability"] is None for p in predictions if p["model"] == "rbf_svm")
    assert report["summaries"]["rbf_svm"]["site_mean"]["brier"] is None


def test_nested_evaluation_rejects_duplicate_participants_and_too_few_groups() -> None:
    features, labels, groups, ids = dataset()
    with pytest.raises(ValueError, match="Duplicate participant"):
        evaluate_baselines(features, labels, groups, ids * 0)
    with pytest.raises(ValueError, match="at least four"):
        evaluate_baselines(features[:18], labels[:18], groups[:18], ids[:18])
