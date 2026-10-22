"""Matched classical and quantum readouts of a fold-fitted graph encoder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantum_neuro_ml.baselines import load_research_dataset
from quantum_neuro_ml.data.cohort import DEFAULT_POLICY, CohortPolicy, write_csv
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models
from quantum_neuro_ml.experiments import (DEFAULT_DATASET, DEFAULT_EXPERIMENTS, bottleneck_plans,
                                          paired_comparisons, validate_configuration)
from quantum_neuro_ml.graph_encoder import SignedGraphEncoder
from quantum_neuro_ml.provenance import digest, recorded_run
from quantum_neuro_ml.representations import BoundedAngles, EdgeFeatures, NuisanceResidualizer


DEFAULT_GRAPH = Path("config/graph-v4.json")


def validate_graph_config(config: dict) -> None:
    fixed = {
        "version": "abide-signed-graph-v4", "representations": ["raw", "residual"], "nodes": 200,
        "node_features": "training_standardized_connectivity_profiles_plus_per_graph_absolute_laplacian_eigenvectors",
        "adjacency": "separate_positive_negative_fisher_weights_normalized_by_one_plus_absolute_degree",
        "encoder": "two_signed_message_layers_mean_max_pool_tanh_embedding", "optimizer": "Adam",
        "loss": "class_balanced_binary_cross_entropy_with_logits",
        "encoder_selection": "fixed_schedule_fit_inside_every_inner_and_outer_training_partition",
        "readouts": ["logistic", "rbf", "polynomial", "quantum"],
        "angle_mapping": "pi_times_tanh_of_standardized_embedding_divided_by_2",
        "training_fraction": 1.0, "dtype": "float32", "hardware": "CPU",
    }
    for key, value in fixed.items():
        if key not in config or config[key] != value:
            raise ValueError(f"Unsupported graph configuration: {key}")
    for key in ("spectral_dimensions", "hidden_units", "embedding_dimensions", "epochs", "batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid graph count: {key}")
    if (not isinstance(config.get("polynomial_solver_max_iter"), int)
            or config["polynomial_solver_max_iter"] < 1
            or not np.isfinite(config.get("polynomial_solver_tolerance", np.nan))
            or config["polynomial_solver_tolerance"] <= 0):
        raise ValueError("Invalid polynomial solver limits")
    if config["spectral_dimensions"] >= config["nodes"] or not 2 <= config["embedding_dimensions"] <= 8:
        raise ValueError("Invalid spectral or embedding dimensions")
    if (not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0
            or not np.isfinite(config["weight_decay"]) or config["weight_decay"] < 0
            or not 0 <= config["dropout"] < 1 or not isinstance(config["seed"], int)
            or not 0 <= config["seed"] < 2 ** 32):
        raise ValueError("Invalid graph training configuration")


def graph_plans(config: dict, kernels: dict, cache: Path) -> list[ModelPlan]:
    d = config["embedding_dimensions"]
    kernel_config = {**kernels, "pca_dimensions": [d], "representations": config["representations"]}
    prototypes = bottleneck_plans(kernel_config, cache)
    memory = Memory(cache, verbose=0)
    plans = []
    for plan in prototypes:
        kind = plan.name.split("_", 1)[0]
        initial = EdgeFeatures() if kind == "raw" else NuisanceResidualizer()
        encoder = SignedGraphEncoder(**{key: config[key] for key in (
            "spectral_dimensions", "hidden_units", "embedding_dimensions", "epochs", "batch_size",
            "learning_rate", "weight_decay", "dropout", "seed")})
        model = plan.estimator.named_steps["model"]
        if getattr(model, "kernel", None) == "poly":
            model.set_params(max_iter=config["polynomial_solver_max_iter"],
                             tol=config["polynomial_solver_tolerance"])
        pipeline = Pipeline([("input", initial), ("graph", encoder), ("scale_embedding", StandardScaler()),
                             ("angles", BoundedAngles()), ("model", model)], memory=memory)
        plans.append(ModelPlan(plan.name.replace(f"pca{d}", f"graph{d}"), pipeline, plan.grid))
    return plans


def snapshot_cache(source: Path, destination: Path, manifest_path: Path) -> dict[str, str]:
    if not source.is_dir() or destination.exists():
        raise ValueError("Encoder cache source must exist and destination must be new")
    shutil.copytree(source, destination)
    manifest = {path.relative_to(destination).as_posix(): digest(path)
                for path in sorted(destination.rglob("*")) if path.is_file()}
    if not manifest:
        raise ValueError("Encoder cache source is empty")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(dataset: Path, policy_path: Path, kernel_path: Path, config_path: Path, output: Path,
        cache_source: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment outputs already exist; choose a new directory")
    inputs = {"dataset": dataset, "dataset_report": dataset.with_name("report.json"),
              "manifest": dataset.with_name("manifest.csv"), "policy": policy_path,
              "kernel_protocol": kernel_path, "graph_protocol": config_path}
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "cache"
    cache_manifest_path = output / "cache-manifest.json"
    if cache_source is not None:
        origin_receipt = cache_source.parent / "run.json"
        origin = json.loads(origin_receipt.read_text(encoding="utf-8"))
        if (origin.get("state") != "failed" or digest(cache_source.parent / "source.zip")
                != origin.get("source_snapshot_sha256")):
            raise ValueError("Encoder cache source must come from an intact failed run")
        snapshot_cache(cache_source, cache_path, cache_manifest_path)
        expected_cache = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        actual_cache = {path.relative_to(cache_path).as_posix(): digest(path)
                        for path in sorted(cache_path.rglob("*")) if path.is_file()}
        if actual_cache != expected_cache:
            raise ValueError("Copied encoder cache does not match its manifest")
        inputs["encoder_cache_manifest"] = cache_manifest_path
        inputs["encoder_cache_origin_receipt"] = origin_receipt
    else:
        cache_path.mkdir()
    with recorded_run(output, {"experiment": "signed-graph-kernels-v4"}, inputs) as receipt:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        kernels = json.loads(kernel_path.read_text(encoding="utf-8"))
        policy = CohortPolicy.load(policy_path)
        validate_configuration(kernels, policy)
        validate_graph_config(config)
        arrays, lineage = load_research_dataset(dataset, policy)
        if arrays["features"].shape[1] != config["nodes"] * (config["nodes"] - 1) // 2:
            raise ValueError("Dataset does not match the graph node configuration")
        features = np.column_stack([arrays["age"], arrays["sex"], arrays["mean_fd"], arrays["features"]])
        data = EvaluationData(features, arrays["labels"], arrays["site_groups"], arrays["subject_ids"])
        plans = graph_plans(config, kernels, cache_path)
        def progress(fold, rows):
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"fold": fold, "predictions": rows}, sort_keys=True) + "\n")
        report, predictions = evaluate_models(data, plans, seed=kernels["seed"], on_fold=progress)
        report.update(experiment_version=config["version"], protocol_configuration=config,
                      kernel_configuration=kernels, dataset_sha256=lineage["dataset_sha256"],
                      policy_sha256=lineage["policy_sha256"], metadata_sha256=lineage["metadata_sha256"],
                      manifest_sha256=lineage["manifest_sha256"])
        if cache_source is not None:
            report["encoder_cache_origin"] = str(cache_source.parent)
            report["encoder_cache_manifest_sha256"] = digest(cache_manifest_path)
        report["paired_comparisons"] = paired_comparisons(report, kernels["seed"])
        write_csv(output / "predictions.csv", predictions,
                  ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"])
        report["predictions_sha256"] = digest(output / "predictions.csv")
        return receipt.finish(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--kernel-config", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--reuse-cache-from", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.policy, args.kernel_config, args.config, args.output_dir, args.reuse_cache_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
