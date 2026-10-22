"""Run the prospectively specified exploratory controls and kernel comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Memory
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quantum_neuro_ml.baselines import load_research_dataset
from quantum_neuro_ml.data.cohort import DEFAULT_POLICY, CohortPolicy, write_csv
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models
from quantum_neuro_ml.provenance import digest, recorded_run
from quantum_neuro_ml.representations import BoundedAngles, CovariateDesign, EdgeFeatures, FirstComponents, NuisanceResidualizer


DEFAULT_EXPERIMENTS = Path("config/experiments-v2.json")
DEFAULT_DATASET = Path("data/cohort/v1-verified/connectivity.npz")


def validate_configuration(config: dict, policy: CohortPolicy) -> None:
    fixed = {
        "cohort_policy": policy.version, "outer_cv": "leave_one_acquisition_group_out",
        "inner_splits": 3, "selection_metric": "balanced_accuracy",
        "covariates": ["age", "age_squared", "sex_code_2", "mean_fd"],
        "residualization": "ordinary_least_squares_all_training_participants_with_intercept",
        "pca_solver": "randomized", "representations": ["raw", "residual"],
        "angle_mapping": "pi_times_tanh_of_standardized_component_divided_by_2",
        "quantum_feature_map": "per_layer_H_then_RY_angles_then_chain_IsingZZ_angle_products_divided_by_pi",
        "quantum_device": "default.qubit", "quantum_shots": None,
        "permutation_strata": "acquisition_group",
        "permutation_model": "full_edge_logistic_with_original_v1_grid",
        "permutation_statistic": "mean_held_out_group_balanced_accuracy",
    }
    for key, value in fixed.items():
        if key not in config or config[key] != value:
            raise ValueError(f"Unsupported {key}: configuration differs from implemented semantics")
    for key in ("logistic_C", "expanded_svm_C", "gamma_factors", "kernel_C"):
        values = config[key]
        if not values or not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"Invalid positive grid: {key}")
    rank = config["pca_rank"]
    if not isinstance(rank, int) or not 1 <= rank <= 8:
        raise ValueError("PCA rank must be between 1 and 8")
    for key in ("pca_dimensions", "low_sample_dimensions"):
        if not config[key] or len(set(config[key])) != len(config[key]) or any(not isinstance(d, int) or not 1 <= d <= rank for d in config[key]):
            raise ValueError(f"Invalid compressed dimensions: {key}")
    if not config["quantum_depths"] or any(depth not in (1, 2) for depth in config["quantum_depths"]):
        raise ValueError("Only circuit depths 1 and 2 are implemented")
    if any(not 0 < fraction < 1 for fraction in config["low_sample_fractions"]):
        raise ValueError("Low-sample fractions must be in (0, 1)")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2 ** 32:
        raise ValueError("Seed must be a nonnegative 32-bit integer")
    if not isinstance(config["label_permutations"], int) or config["label_permutations"] < 1:
        raise ValueError("Specify a positive number of label permutations")


def _logistic(seed: int, *, dual: bool) -> LogisticRegression:
    return LogisticRegression(solver="liblinear", dual=dual, class_weight="balanced", max_iter=5000, random_state=seed)


def _pipeline(kind: str, model: object, memory: Memory, config: dict, dimensions: int | None = None) -> Pipeline:
    initial = {"raw": EdgeFeatures, "residual": NuisanceResidualizer, "covariates": CovariateDesign}[kind]()
    steps = [("input", initial), ("scale_edges", StandardScaler())]
    if dimensions is not None:
        steps += [
            ("pca", PCA(n_components=config["pca_rank"], svd_solver=config["pca_solver"],
                        iterated_power=config["pca_iterated_power"], random_state=config["seed"])),
            ("components", FirstComponents(dimensions)), ("scale_components", StandardScaler()),
            ("angles", BoundedAngles()),
        ]
    return Pipeline([*steps, ("model", model)], memory=memory)


def control_plans(config: dict, edges: int, cache: Path) -> list[ModelPlan]:
    memory = Memory(cache, verbose=0)
    plans = []
    for kind in ("covariates", "raw", "residual"):
        plans.append(ModelPlan(f"{kind}_logistic", _pipeline(kind, _logistic(config["seed"], dual=kind != "covariates"), memory, config),
                               {"model__C": config["logistic_C"]}))
        p = 4 if kind == "covariates" else edges
        plans.append(ModelPlan(f"{kind}_rbf_expanded", _pipeline(kind, SVC(kernel="rbf", class_weight="balanced"), memory, config),
                               {"model__C": config["expanded_svm_C"], "model__gamma": [factor / p for factor in config["gamma_factors"]]}))
    return plans


def bottleneck_plans(config: dict, cache: Path, fraction: float = 1.0) -> list[ModelPlan]:
    from quantum_neuro_ml.quantum_kernels import FidelityKernelSVC
    memory = Memory(cache, verbose=0)
    dimensions = config["pca_dimensions"] if fraction == 1.0 else config["low_sample_dimensions"]
    plans = []
    for kind in config["representations"]:
        for d in dimensions:
            classifiers = [
                ("logistic", _logistic(config["seed"], dual=False), {"model__C": config["logistic_C"]}),
                ("rbf", SVC(kernel="rbf", class_weight="balanced"),
                 {"model__C": config["kernel_C"], "model__gamma": [factor / d for factor in config["gamma_factors"]]}),
                ("polynomial", SVC(kernel="poly", class_weight="balanced", gamma=1 / d, coef0=config["polynomial_coef0"]),
                 {"model__C": config["kernel_C"], "model__degree": config["polynomial_degrees"]}),
                ("quantum", FidelityKernelSVC(), {"model__C": config["kernel_C"], "model__depth": config["quantum_depths"]}),
            ]
            for family, model, grid in classifiers:
                plans.append(ModelPlan(f"{kind}_pca{d}_{family}", _pipeline(kind, model, memory, config, d), grid))
    return plans


def paired_comparisons(report: dict, seed: int) -> dict:
    folds = report["folds"]
    result = {}
    quantum_names = sorted(name for name in report["summaries"] if name.endswith("_quantum"))
    for quantum in quantum_names:
        prefix = quantum.removesuffix("_quantum")
        q_folds = {f["held_out_group"]: f for f in folds if f["model"] == quantum}
        for family in ("logistic", "rbf", "polynomial"):
            other = f"{prefix}_{family}"
            classical = {f["held_out_group"]: f for f in folds if f["model"] == other}
            deltas = {}
            for group in sorted(q_folds):
                q, c = q_folds[group], classical[group]
                if q["training_subject_ids_sha256"] != c["training_subject_ids_sha256"]:
                    raise ValueError("Matched models received different training participants")
                for key in ("training_representation_sha256", "test_representation_sha256"):
                    if q["diagnostics"][key] != c["diagnostics"][key]:
                        raise ValueError("Matched models received different compressed representations")
            for metric in ("balanced_accuracy", "auroc"):
                values = np.array([q_folds[g]["metrics"][metric] - classical[g]["metrics"][metric] for g in sorted(q_folds)])
                rng = np.random.default_rng(seed)
                bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
                deltas[metric] = {"mean_quantum_minus_classical": float(values.mean()),
                                  "paired_site_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                                  "by_group": dict(zip(sorted(q_folds), values.tolist()))}
            result[f"{quantum}_minus_{other}"] = deltas
    return result


def run_experiment(dataset: Path, policy_path: Path, config_path: Path, output: Path,
                   *, phase: str = "controls", fraction: float = 1.0) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment outputs already exist; choose a new directory")
    if phase not in ("controls", "bottleneck"):
        raise ValueError("Unsupported experimental phase")
    settings = {"experiment": "exploratory-followup", "phase": phase, "training_fraction": fraction}
    inputs = {"dataset": dataset, "dataset_report": dataset.with_name("report.json"),
              "manifest": dataset.with_name("manifest.csv"), "policy": policy_path, "protocol": config_path}
    with recorded_run(output, settings, inputs) as run:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        policy = CohortPolicy.load(policy_path)
        validate_configuration(config, policy)
        if fraction not in [1.0, *config["low_sample_fractions"]] or (phase == "controls" and fraction != 1):
            raise ValueError("Training fraction is not specified for this experiment")
        arrays, dataset_report = load_research_dataset(dataset, policy)
        X = np.column_stack([arrays["age"], arrays["sex"], arrays["mean_fd"], arrays["features"]])
        data = EvaluationData(X, arrays["labels"], arrays["site_groups"], arrays["subject_ids"])
        plans = control_plans(config, arrays["features"].shape[1], output / "cache") if phase == "controls" else bottleneck_plans(config, output / "cache", fraction)
        def progress(fold: dict, rows: list[dict]) -> None:
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"fold": fold, "predictions": rows}, sort_keys=True) + "\n")
        report, predictions = evaluate_models(data, plans, seed=config["seed"], training_fraction=fraction, on_fold=progress)
        report.update(experiment_version=config["version"], phase=phase, protocol_sha256=digest(config_path),
                      protocol_configuration=config,
                      dataset_sha256=dataset_report["dataset_sha256"], policy_sha256=dataset_report["policy_sha256"],
                      manifest_sha256=dataset_report["manifest_sha256"], metadata_sha256=dataset_report["metadata_sha256"])
        if phase == "bottleneck":
            report["paired_comparisons"] = paired_comparisons(report, config["seed"])
        write_csv(output / "predictions.csv", predictions,
                  ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"])
        report["predictions_sha256"] = digest(output / "predictions.csv")
        return run.finish(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--phase", choices=["controls", "bottleneck"], default="controls")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_experiment(args.dataset, args.policy, args.config, args.output_dir, phase=args.phase, fraction=args.fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
