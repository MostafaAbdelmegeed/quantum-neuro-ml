"""Within-acquisition-group label permutations of the full nested reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantum_neuro_ml.baselines import _model_spec, load_research_dataset
from quantum_neuro_ml.data.cohort import DEFAULT_POLICY, CohortPolicy, write_csv
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models
from quantum_neuro_ml.experiments import DEFAULT_DATASET, DEFAULT_EXPERIMENTS, validate_configuration
from quantum_neuro_ml.provenance import digest, recorded_run


_DATA: EvaluationData | None = None


def permuted_labels(labels: np.ndarray, groups: np.ndarray, seed: int, trial: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial + 1]))
    result = labels.copy()
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        result[indices] = labels[rng.permutation(indices)]
    return result


def _initialize(dataset: str, policy: str) -> None:
    global _DATA
    arrays, _ = load_research_dataset(Path(dataset), CohortPolicy.load(Path(policy)))
    _DATA = EvaluationData(arrays["features"], arrays["labels"], arrays["site_groups"], arrays["subject_ids"])


def _reference_plan() -> ModelPlan:
    estimator, grid = _model_spec("logistic", 19900)
    return ModelPlan("logistic", Pipeline([("scale", StandardScaler()), ("model", estimator)]), grid)


def _trial(trial: int, seed: int) -> dict:
    started = time.perf_counter()
    if _DATA is None:
        raise RuntimeError("Permutation worker was not initialized")
    labels = permuted_labels(_DATA.labels, _DATA.groups, seed, trial)
    for group in set(_DATA.groups):
        mask = _DATA.groups == group
        if np.bincount(labels[mask], minlength=2).tolist() != np.bincount(_DATA.labels[mask], minlength=2).tolist():
            raise RuntimeError("Permutation changed a site-specific class count")
    data = EvaluationData(_DATA.features, labels, _DATA.groups, _DATA.subject_ids)
    report, _ = evaluate_models(data, [_reference_plan()], seed=seed, verbose=False)
    mean = report["summaries"]["logistic"]["site_mean"]
    return {"trial": trial, "balanced_accuracy": mean["balanced_accuracy"], "auroc": mean["auroc"],
            "permuted_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "seconds": time.perf_counter() - started,
            "by_group": {f["held_out_group"]: f["metrics"] for f in report["folds"]}}


def run_permutations(dataset: Path, policy_path: Path, config_path: Path, reference: Path,
                     output: Path, workers: int = 2) -> dict:
    if workers not in range(1, 5):
        raise ValueError("Choose one to four permutation workers")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Permutation outputs already exist; choose a new directory")
    inputs = {"dataset": dataset, "dataset_report": dataset.with_name("report.json"),
              "manifest": dataset.with_name("manifest.csv"), "policy": policy_path,
              "protocol": config_path, "reference_report": reference,
              "reference_predictions": reference.with_name("predictions.csv")}
    with recorded_run(output, {"experiment": "within-site-label-permutations", "workers": workers}, inputs) as run:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        policy = CohortPolicy.load(policy_path)
        validate_configuration(config, policy)
        _initialize(str(dataset), str(policy_path))
        expected = json.loads(reference.read_text(encoding="utf-8"))
        if expected["dataset_sha256"] != digest(dataset):
            raise ValueError("Permutation reference used a different dataset")
        observed_report, observed_rows = evaluate_models(_DATA, [_reference_plan()], seed=config["seed"])
        observed = observed_report["summaries"]["logistic"]["site_mean"]["balanced_accuracy"]
        if not np.isclose(observed, expected["summaries"]["logistic"]["site_mean"]["balanced_accuracy"], atol=1e-12, rtol=0):
            raise ValueError("Shared evaluator did not reproduce the v1 observed statistic")
        with reference.with_name("predictions.csv").open(newline="") as stream:
            expected_rows = {int(row["subject_id"]): row for row in csv.DictReader(stream) if row["model"] == "logistic"}
        for row in observed_rows:
            old = expected_rows[row["subject_id"]]
            if row["prediction"] != int(old["prediction"]) or not np.isclose(row["score"], float(old["score"]), atol=1e-12, rtol=0):
                raise ValueError("Shared evaluator did not reproduce v1 held-out predictions")
        write_csv(output / "observed-predictions.csv", observed_rows,
                  ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"])
        count = config["label_permutations"]
        results = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                 initargs=(str(dataset.resolve()), str(policy_path.resolve()))) as pool:
            futures = [pool.submit(_trial, trial, config["seed"]) for trial in range(count)]
            for future in as_completed(futures):
                value = future.result()
                results.append(value)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, sort_keys=True) + "\n")
                print(f"Permutation {len(results)}/{count}: balanced_accuracy={value['balanced_accuracy']:.3f}", flush=True)
        results.sort(key=lambda r: r["trial"])
        scores = np.array([r["balanced_accuracy"] for r in results])
        exceedances = int(np.sum(scores >= observed - 1e-12))
        report = {
            "purpose": "within_site_nested_permutation_test", "seed": config["seed"], "permutations": count,
            "workers": workers, "observed_balanced_accuracy": observed, "null_mean": float(scores.mean()),
            "null_percentiles": np.quantile(scores, [0.025, 0.5, 0.975]).tolist(),
            "exceedances": exceedances, "empirical_one_sided_p": (1 + exceedances) / (count + 1),
            "protocol_sha256": digest(config_path), "dataset_sha256": digest(dataset),
            "policy_sha256": policy.sha256, "observed": observed_report, "trials": results,
        }
        return run.finish(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--reference", type=Path, default=Path("results/classical-v1-verified/report.json"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_permutations(args.dataset, args.policy, args.config, args.reference, args.output_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
