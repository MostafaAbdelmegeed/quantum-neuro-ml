import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("result_audit", Path("scripts/audit_results.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_independent_metrics_handle_tied_scores():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    result = audit.independent_metrics(labels, predictions, scores)
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["auroc"] == pytest.approx(0.875)
    assert audit.independent_metrics(labels, np.zeros(4, dtype=int), np.ones(4))["auroc"] == 0.5
    with pytest.raises(ValueError, match="binary"):
        audit.independent_metrics(labels, np.array([0, 0, 2, 1]), scores)


def test_audit_sampling_excludes_held_out_and_matches_fraction():
    ids = np.arange(48, dtype=np.int64)
    labels = np.tile([0, 1], 24)
    groups = np.repeat(["A", "B", "C", "D"], 12)
    a = audit.training_indices(ids, labels, groups, "A", 0.25, 42)
    b = audit.training_indices(ids, labels, groups, "A", 0.5, 42)
    assert set(a) <= set(b)
    assert "A" not in set(groups[a])
    assert len(a) == 12 and len(b) == 18


def test_duplicate_fold_records_are_rejected(tmp_path, monkeypatch):
    import csv

    ids = np.arange(8)
    labels, groups = np.tile([0, 1], 4), np.repeat(["A", "B", "C", "D"], 2)
    with (tmp_path / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "subject_id"])
        writer.writeheader()
        writer.writerows({"model": "logistic", "subject_id": int(i)} for i in ids)
    folds = [{"model": "logistic", "held_out_group": group} for group in sorted(set(groups))]
    report = {"predictions_sha256": "test", "summaries": {"logistic": {}}, "folds": folds + [folds[0]]}
    monkeypatch.setattr(audit, "verify_receipt", lambda directory: report)
    monkeypatch.setattr(audit, "digest", lambda path: "test")
    with pytest.raises(ValueError, match="Fold inventory"):
        audit.audit_predictions(tmp_path, ids, labels, groups)


def test_permutation_audit_anchors_observed_and_null_summaries(tmp_path, monkeypatch):
    import csv
    import hashlib
    import json

    ids = np.arange(48)
    labels, groups = np.tile([0, 1], 24), np.repeat(["A", "B", "C", "D"], 12)
    reference = tmp_path / "reference"
    reference.mkdir()
    reference_report = {"summaries": {"logistic": {"site_mean": {"balanced_accuracy": 1.0}}}}
    (reference / "report.json").write_text(json.dumps(reference_report))
    fields = ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"]
    rows = [dict(zip(fields, ["logistic", int(i), str(g), int(y), int(y), float(y), None]))
            for i, y, g in zip(ids, labels, groups)]
    for path in (reference / "predictions.csv", tmp_path / "observed-predictions.csv"):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    inputs = {"reference_report": {"path": str(reference / "report.json")},
              "reference_predictions": {"path": str(reference / "predictions.csv")}}
    (tmp_path / "run.json").write_text(json.dumps({"inputs": inputs}))
    trials = []
    for trial in range(99):
        rng = np.random.default_rng(np.random.SeedSequence([42, trial + 1]))
        shuffled = labels.copy()
        for group in sorted(set(groups)):
            indices = np.flatnonzero(groups == group)
            shuffled[indices] = labels[rng.permutation(indices)]
        trials.append({"trial": trial, "balanced_accuracy": 0.5, "auroc": 0.5,
                       "permuted_labels_sha256": hashlib.sha256(shuffled.tobytes()).hexdigest(),
                       "by_group": {g: {"balanced_accuracy": 0.5, "auroc": 0.5} for g in set(groups)}})
    observed = {"folds": [{"held_out_group": g, "metrics": {"balanced_accuracy": 1.0, "auroc": 1.0}} for g in sorted(set(groups))],
                "summaries": {"logistic": {"site_mean": {"balanced_accuracy": 1.0, "auroc": 1.0}}}}
    report = {"seed": 42, "trials": trials, "permutations": 99, "observed_balanced_accuracy": 1.0,
              "observed": observed, "exceedances": 0, "empirical_one_sided_p": 0.01,
              "null_mean": 0.5, "null_percentiles": [0.5, 0.5, 0.5]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(audit, "verify_receipt", lambda directory: report)
    assert audit.audit_permutations(tmp_path, ids, labels, groups)["unique_target_vectors"] == 99
    report["observed_balanced_accuracy"] = 0.99
    with pytest.raises(ValueError, match="recorded reference"):
        audit.audit_permutations(tmp_path, ids, labels, groups)
    report["observed_balanced_accuracy"] = 1.0
    observed["summaries"]["logistic"]["site_mean"]["balanced_accuracy"] = 0.69
    with pytest.raises(ValueError, match="Nested observed"):
        audit.audit_permutations(tmp_path, ids, labels, groups)
    observed["summaries"]["logistic"]["site_mean"]["balanced_accuracy"] = 1.0
    report["null_percentiles"] = [9, 9, 9]
    with pytest.raises(AssertionError):
        audit.audit_permutations(tmp_path, ids, labels, groups)
