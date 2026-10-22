"""Evaluate fixed classical baselines with nested acquisition-group folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quantum_neuro_ml.data.cohort import DEFAULT_OUTPUT, DEFAULT_POLICY, CohortPolicy, write_csv
from quantum_neuro_ml.data.splits import acquisition_group
from quantum_neuro_ml.data.abide import build_derivative_url
from quantum_neuro_ml.provenance import recorded_run
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models, validate_arrays as _validate_arrays


SEED = 42
MODEL_NAMES = ("dummy", "logistic", "rbf_svm")


def _model_spec(name: str, features: int) -> tuple[object, dict[str, list]]:
    if name == "dummy":
        return DummyClassifier(strategy="prior"), {}
    if name == "logistic":
        return LogisticRegression(solver="liblinear", dual=True, class_weight="balanced",
                                  max_iter=5000, random_state=SEED), {"model__C": [0.001, 0.01, 0.1, 1.0]}
    if name == "rbf_svm":
        return SVC(kernel="rbf", class_weight="balanced"), {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": [factor / features for factor in (0.1, 1.0, 10.0)],
        }
    raise ValueError(f"Unknown model: {name}")


def evaluate_baselines(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, subject_ids: np.ndarray,
    *, models: Sequence[str] = MODEL_NAMES,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run the frozen v1 plans through the shared nested evaluator."""
    if not models or len(set(models)) != len(models) or set(models) - set(MODEL_NAMES):
        raise ValueError("Choose distinct supported models")
    plans = []
    for name in models:
        estimator, grid = _model_spec(name, features.shape[1])
        pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        plans.append(ModelPlan(name, pipeline, grid))
    report, predictions = evaluate_models(EvaluationData(features, labels, groups, subject_ids), plans, seed=SEED)
    report["purpose"] = "classical_reference_evaluation"
    report["scaling"] = "StandardScaler with mean centering and unit variance, fitted inside each fold"
    return report, predictions


def load_research_dataset(path: Path, policy: CohortPolicy) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    report = json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
    if report.get("purpose") != "research_dataset" or report.get("policy_sha256") != policy.sha256:
        raise ValueError("Dataset was not built with the selected research policy")
    if report.get("metadata_sha256") != policy.metadata_sha256:
        raise ValueError("Dataset metadata fingerprint differs from the frozen policy")
    if report.get("derivative") != "cpac/filt_noglobal/rois_cc200":
        raise ValueError("Dataset derivative differs from the frozen configuration")
    manifest_path = path.with_name("manifest.csv")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != report.get("manifest_sha256"):
        raise ValueError("Research manifest checksum mismatch")
    run_path = path.with_name("run.json")
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("state") != "complete" or run.get("report_sha256") != hashlib.sha256(path.with_name("report.json").read_bytes()).hexdigest():
            raise ValueError("Research run is incomplete or its completion receipt differs")
        snapshot = path.with_name("source.zip")
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != run.get("source_snapshot_sha256"):
            raise ValueError("Research source snapshot differs from its receipt")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("dataset_sha256"):
        raise ValueError("Research dataset checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _validate_arrays(arrays["features"], arrays["labels"], arrays["site_groups"], arrays["subject_ids"])
    n = len(arrays["labels"])
    for key in ("site_ids", "age", "sex", "mean_fd"):
        if key not in arrays or arrays[key].shape != (n,):
            raise ValueError(f"Missing or misaligned research array: {key}")
    if not all(np.isfinite(arrays[key]).all() for key in ("age", "sex", "mean_fd")):
        raise ValueError("Non-finite research covariates")
    if np.any(arrays["age"] <= 0) or not np.isin(arrays["sex"], [1, 2]).all():
        raise ValueError("Research covariates violate age or sex inclusion rules")
    if np.any(arrays["mean_fd"] < 0) or np.any(arrays["mean_fd"] >= policy.max_mean_fd):
        raise ValueError("Research motion violates the frozen threshold")
    if not np.issubdtype(arrays["subject_ids"].dtype, np.integer) or np.any(arrays["subject_ids"] <= 0):
        raise ValueError("Research participant IDs must be positive integers")
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != n:
        raise ValueError("Research manifest and dataset have different row counts")
    try:
        expected = {
            "subject_ids": np.array([int(row["subject_id"]) for row in rows]),
            "labels": np.array([int(row["diagnosis"]) == 1 for row in rows], dtype=int),
            "site_ids": np.array([row["site_id"] for row in rows]),
            "site_groups": np.array([row["acquisition_group"] for row in rows]),
            "age": np.array([float(row["age_at_scan"]) for row in rows]),
            "sex": np.array([int(row["sex"]) for row in rows]),
            "mean_fd": np.array([float(row["mean_fd"]) for row in rows]),
        }
        if any(int(row["diagnosis"]) not in (1, 2) or row["derivative_url"] != build_derivative_url(row["file_id"]) for row in rows):
            raise ValueError("Invalid manifest diagnosis or derivative URL")
        if len({row["file_id"] for row in rows}) != n:
            raise ValueError("Duplicate manifest file IDs")
    except (KeyError, TypeError) as error:
        raise ValueError("Research manifest lacks required fields") from error
    if any(not np.array_equal(arrays[key], values) for key, values in expected.items()):
        raise ValueError("Research manifest rows do not align with the dataset arrays")
    if not np.array_equal(arrays["site_groups"], [acquisition_group(str(site)) for site in arrays["site_ids"]]):
        raise ValueError("Dataset acquisition groups do not match its original site labels")
    if arrays["features"].shape[1] != 19900 or report.get("feature_count") != 19900:
        raise ValueError("Research baseline expects 19,900 CC200 edges")
    for group in np.unique(arrays["site_groups"]):
        selected = arrays["labels"][arrays["site_groups"] == group]
        if len(selected) < policy.min_site_subjects or any(np.sum(selected == label) < policy.min_site_per_diagnosis for label in (0, 1)):
            raise ValueError("Dataset violates the frozen site-size policy")
    return arrays, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "connectivity.npz")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=Path("results/classical-v1"))
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Model outputs already exist; use a new output directory")
    configuration = {"experiment": "classical-reference-v1", "seed": SEED,
                     "grids": {name: _model_spec(name, 19900)[1] for name in MODEL_NAMES}}
    inputs = {"dataset": args.dataset, "dataset_report": args.dataset.with_name("report.json"),
              "manifest": args.dataset.with_name("manifest.csv"), "policy": args.policy}
    with recorded_run(args.output_dir, configuration, inputs) as run:
        policy = CohortPolicy.load(args.policy)
        arrays, dataset_report = load_research_dataset(args.dataset, policy)
        report, predictions = evaluate_baselines(arrays["features"], arrays["labels"], arrays["site_groups"], arrays["subject_ids"])
        report.update({"policy_sha256": policy.sha256, "dataset_sha256": dataset_report["dataset_sha256"],
                       "metadata_sha256": dataset_report["metadata_sha256"],
                       "manifest_sha256": dataset_report["manifest_sha256"]})
        write_csv(args.output_dir / "predictions.csv", predictions,
                  ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"])
        report["predictions_sha256"] = hashlib.sha256((args.output_dir / "predictions.csv").read_bytes()).hexdigest()
        run.finish(report)
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
