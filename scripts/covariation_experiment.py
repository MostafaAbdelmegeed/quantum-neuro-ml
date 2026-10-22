"""Run nested cross-site models on training-fold edge co-variation modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Memory
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quantum_neuro_ml.baselines import load_research_dataset
from quantum_neuro_ml.covariation import EdgeCovariationGroups
from quantum_neuro_ml.data.cohort import DEFAULT_POLICY, CohortPolicy, write_csv
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models
from quantum_neuro_ml.experiments import DEFAULT_DATASET, DEFAULT_EXPERIMENTS
from quantum_neuro_ml.provenance import digest, recorded_run
from quantum_neuro_ml.representations import BoundedAngles, EdgeFeatures, NuisanceResidualizer


DEFAULT_PROTOCOL = Path("config/covariation-v1.json")


def validate_protocol(config: dict, policy: CohortPolicy, representation: dict) -> None:
    fixed = {
        "version": "abide-edge-covariation-v1", "cohort_policy": policy.version,
        "base_representation_protocol": representation["version"],
        "outer_cv": "leave_one_acquisition_group_out", "inner_splits": 3,
        "selection_metric": "balanced_accuracy", "representations": ["raw", "residual"],
        "candidate_edges": 256, "dimensions": [4, 8, 16],
        "clustering": "average_linkage_on_one_minus_absolute_training_subject_edge_correlation",
        "module_orientation": "sign_align_to_training_co_variation_medoid",
        "module_summary": "mean_training_standardized_signed_edge_value",
        "labels_used_for_module_learning": False, "quantum_dimensions": [4],
        "quantum_depths": [1, 2], "quantum_device": "default.qubit", "quantum_shots": None,
        "quantum_feature_map": "per_layer_H_then_RY_angles_then_chain_IsingZZ_angle_products_divided_by_pi",
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Unsupported co-variation protocol field: {key}")
    if (not isinstance(config.get("candidate_edges"), int) or config["candidate_edges"] < 16
            or config["candidate_edges"] > 19_900
            or config.get("dimensions") != sorted(set(config["dimensions"]))
            or any(not isinstance(value, int) or not 2 <= value <= 8 for value in config["quantum_dimensions"])
            or config["seed"] != representation["seed"]):
        raise ValueError("Invalid module dimensions, candidate count, or seed alignment")
    for key in ("logistic_C", "kernel_C", "gamma_factors"):
        if not config.get(key) or any(not np.isfinite(value) or value <= 0 for value in config[key]):
            raise ValueError(f"Invalid positive grid: {key}")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("Invalid random seed")


def _logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(solver="liblinear", dual=False, class_weight="balanced",
                              max_iter=5000, random_state=seed)


def module_plans(config: dict, cache: Path) -> list[ModelPlan]:
    from quantum_neuro_ml.quantum_kernels import FidelityKernelSVC

    memory = Memory(cache, verbose=0)
    plans = []
    for kind in config["representations"]:
        for dimensions in config["dimensions"]:
            base = EdgeFeatures() if kind == "raw" else NuisanceResidualizer()
            transform = EdgeCovariationGroups(dimensions=dimensions,
                                               candidate_edges=config["candidate_edges"])
            classical_steps = [("input", base), ("modules", transform), ("scale_modules", StandardScaler())]
            classifiers = [
                ("logistic", _logistic(config["seed"]),
                 {"model__C": config["logistic_C"]}),
                ("rbf", SVC(kernel="rbf", class_weight="balanced"),
                 {"model__C": config["kernel_C"],
                  "model__gamma": [factor / dimensions for factor in config["gamma_factors"]]}),
            ]
            for family, model, grid in classifiers:
                pipeline = Pipeline([*classical_steps, ("model", model)], memory=memory)
                plans.append(ModelPlan(f"{kind}_covar{dimensions}_{family}", pipeline, grid))
            if dimensions in config["quantum_dimensions"]:
                for depth in config["quantum_depths"]:
                    quantum_steps = [
                        ("input", EdgeFeatures() if kind == "raw" else NuisanceResidualizer()),
                        ("modules", EdgeCovariationGroups(dimensions=dimensions,
                                                          candidate_edges=config["candidate_edges"])),
                        ("scale_modules", StandardScaler()), ("angles", BoundedAngles()),
                        ("model", FidelityKernelSVC(depth=depth)),
                    ]
                    plans.append(ModelPlan(
                        f"{kind}_covar{dimensions}_quantum_depth{depth}",
                        Pipeline(quantum_steps, memory=memory),
                        {"model__C": config["kernel_C"]},
                    ))
    return plans


def paired_quantum_differences(report: dict, seed: int) -> dict:
    results = {}
    for kind in ("raw", "residual"):
        for depth in (1, 2):
            quantum_name = f"{kind}_covar4_quantum_depth{depth}"
            quantum = {f["held_out_group"]: f for f in report["folds"] if f["model"] == quantum_name}
            for family in ("logistic", "rbf"):
                classical_name = f"{kind}_covar4_{family}"
                classical = {f["held_out_group"]: f for f in report["folds"] if f["model"] == classical_name}
                if set(quantum) != set(classical) or not quantum:
                    raise ValueError("Quantum and classical comparisons must share all held-out groups")
                for group in quantum:
                    if (quantum[group]["training_subject_ids_sha256"]
                            != classical[group]["training_subject_ids_sha256"]):
                        raise ValueError("Paired models received different training participants")
                    q_modules = quantum[group]["diagnostics"]["modules"]
                    c_modules = classical[group]["diagnostics"]["modules"]
                    if q_modules["module_assignment_sha256"] != c_modules["module_assignment_sha256"]:
                        raise ValueError("Quantum and classical models received different learned modules")
                deltas = {}
                for metric in ("balanced_accuracy", "auroc"):
                    values = np.array([quantum[group]["metrics"][metric] - classical[group]["metrics"][metric]
                                       for group in sorted(quantum)])
                    rng = np.random.default_rng(seed)
                    bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
                    deltas[metric] = {
                        "mean_quantum_minus_classical": float(values.mean()),
                        "paired_site_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                        "by_group": dict(zip(sorted(quantum), values.tolist())),
                    }
                results[f"{quantum_name}_minus_{classical_name}"] = deltas
    return results


def run(dataset: Path, policy_path: Path, representation_path: Path,
        protocol_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment outputs already exist; choose a new directory")
    inputs = {"dataset": dataset, "dataset_report": dataset.with_name("report.json"),
              "manifest": dataset.with_name("manifest.csv"), "policy": policy_path,
              "base_representation_protocol": representation_path,
              "covariation_protocol": protocol_path}
    with recorded_run(output, {"experiment": "edge-covariation-modules-v1"}, inputs) as receipt:
        config = json.loads(protocol_path.read_text(encoding="utf-8"))
        representation = json.loads(representation_path.read_text(encoding="utf-8"))
        policy = CohortPolicy.load(policy_path)
        validate_protocol(config, policy, representation)
        arrays, lineage = load_research_dataset(dataset, policy)
        features = np.column_stack([arrays["age"], arrays["sex"], arrays["mean_fd"], arrays["features"]])
        data = EvaluationData(features, arrays["labels"], arrays["site_groups"], arrays["subject_ids"])
        plans = module_plans(config, output / "cache")

        def progress(fold, rows):
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"fold": fold, "predictions": rows}, sort_keys=True) + "\n")

        report, predictions = evaluate_models(data, plans, seed=config["seed"], on_fold=progress)
        report.update(experiment_version=config["version"], protocol_configuration=config,
                      base_representation_configuration=representation,
                      dataset_sha256=lineage["dataset_sha256"], policy_sha256=lineage["policy_sha256"],
                      manifest_sha256=lineage["manifest_sha256"], metadata_sha256=lineage["metadata_sha256"])
        report["paired_comparisons"] = paired_quantum_differences(report, config["seed"])
        write_csv(output / "predictions.csv", predictions,
                  ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"])
        report["predictions_sha256"] = digest(output / "predictions.csv")
        return receipt.finish(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--representation-protocol", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.policy, args.representation_protocol, args.protocol, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
