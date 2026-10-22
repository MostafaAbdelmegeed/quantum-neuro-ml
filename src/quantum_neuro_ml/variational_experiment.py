"""Nested, matched variational and MLP readouts on four PCA coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Memory

from quantum_neuro_ml.baselines import load_research_dataset
from quantum_neuro_ml.data.cohort import DEFAULT_POLICY, CohortPolicy, write_csv
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models
from quantum_neuro_ml.experiments import DEFAULT_DATASET, DEFAULT_EXPERIMENTS, _pipeline, validate_configuration
from quantum_neuro_ml.provenance import digest, recorded_run
from quantum_neuro_ml.variational import TrainableReadout


DEFAULT_VARIATIONAL = Path("config/variational-v3.json")


def validate_readout_config(config: dict) -> None:
    fixed = {
        "version": "abide-variational-v3", "representation_protocol": "abide-exploratory-v2",
        "representations": ["raw", "residual"], "dimensions": 4, "optimizer": "Adam",
        "loss": "class_balanced_binary_cross_entropy_with_logits", "quantum_encoding": "initial_RY_angles",
        "quantum_layer": "trainable_RZ_RY_then_directed_CNOT_ring",
        "quantum_readout": "all_PauliZ_expectations_then_trainable_linear_logit",
        "quantum_device": "default.qubit", "quantum_shots": None, "quantum_gradient": "backprop",
        "classical_activation": "tanh", "dtype": "float64", "hardware": "CPU",
        "early_stopping": False, "training_fraction": 1.0,
    }
    for key, value in fixed.items():
        if key not in config or config[key] != value:
            raise ValueError(f"Unsupported variational configuration: {key}")
    if (not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"])
            or any(not isinstance(s, int) or not 0 <= s < 2 ** 32 for s in config["seeds"])):
        raise ValueError("Invalid readout seeds")
    if not config["learning_rates"] or any(not np.isfinite(v) or v <= 0 for v in config["learning_rates"]):
        raise ValueError("Invalid learning rates")
    for key in ("epochs", "batch_size", "classical_hidden_units"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["quantum_depth"] not in (1, 2) or not np.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("Invalid depth or weight decay")


def readout_plans(config: dict, representation_config: dict, cache: Path) -> list[ModelPlan]:
    memory = Memory(cache, verbose=0)
    plans = []
    for kind in config["representations"]:
        for seed in config["seeds"]:
            for family in ("mlp", "quantum"):
                model = TrainableReadout(family=family, epochs=config["epochs"], batch_size=config["batch_size"],
                                         weight_decay=config["weight_decay"], seed=seed, depth=config["quantum_depth"],
                                         hidden_units=config["classical_hidden_units"])
                name = f"{kind}_pca4_seed{seed}_{family}"
                plans.append(ModelPlan(name, _pipeline(kind, model, memory, representation_config, config["dimensions"]),
                                       {"model__learning_rate": config["learning_rates"]}))
    return plans


def paired_difference(report: dict, first: str, second: str, seed: int = 42) -> dict:
    a = {f["held_out_group"]: f for f in report["folds"] if f["model"] == first}
    b = {f["held_out_group"]: f for f in report["folds"] if f["model"] == second}
    if set(a) != set(b) or not a:
        raise ValueError("Paired comparison needs the same held-out acquisition groups")
    for group in a:
        if a[group]["training_subject_ids_sha256"] != b[group]["training_subject_ids_sha256"]:
            raise ValueError("Paired models received different training participants")
        for key in ("training_representation_sha256", "test_representation_sha256"):
            if a[group]["diagnostics"][key] != b[group]["diagnostics"][key]:
                raise ValueError("Paired models received different input representations")
    result = {}
    for metric in ("balanced_accuracy", "auroc"):
        values = np.array([a[g]["metrics"][metric] - b[g]["metrics"][metric] for g in sorted(a)])
        rng = np.random.default_rng(seed)
        bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
        result[metric] = {"mean_first_minus_second": float(values.mean()),
                          "paired_site_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                          "by_group": dict(zip(sorted(a), values.tolist()))}
    return result


def run(dataset: Path, policy_path: Path, representation_path: Path, config_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment outputs already exist; choose a new directory")
    inputs = {"dataset": dataset, "dataset_report": dataset.with_name("report.json"),
              "manifest": dataset.with_name("manifest.csv"), "policy": policy_path,
              "representation_protocol": representation_path, "readout_protocol": config_path}
    with recorded_run(output, {"experiment": "variational-readouts-v3"}, inputs) as receipt:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        representation = json.loads(representation_path.read_text(encoding="utf-8"))
        policy = CohortPolicy.load(policy_path)
        validate_configuration(representation, policy)
        validate_readout_config(config)
        if representation["version"] != config["representation_protocol"]:
            raise ValueError("Readout and representation protocols differ")
        arrays, lineage = load_research_dataset(dataset, policy)
        X = np.column_stack([arrays["age"], arrays["sex"], arrays["mean_fd"], arrays["features"]])
        data = EvaluationData(X, arrays["labels"], arrays["site_groups"], arrays["subject_ids"])
        plans = readout_plans(config, representation, output / "cache")
        def progress(fold, rows):
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"fold": fold, "predictions": rows}, sort_keys=True) + "\n")
        report, predictions = evaluate_models(data, plans, seed=representation["seed"], on_fold=progress)
        report.update(experiment_version=config["version"], protocol_configuration=config,
                      representation_configuration=representation, dataset_sha256=lineage["dataset_sha256"],
                      policy_sha256=lineage["policy_sha256"], manifest_sha256=lineage["manifest_sha256"],
                      metadata_sha256=lineage["metadata_sha256"])
        report["paired_comparisons"] = {}
        for kind in config["representations"]:
            for seed in config["seeds"]:
                prefix = f"{kind}_pca4_seed{seed}"
                report["paired_comparisons"][prefix + "_quantum_minus_mlp"] = paired_difference(
                    report, prefix + "_quantum", prefix + "_mlp", representation["seed"])
        write_csv(output / "predictions.csv", predictions,
                  ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"])
        report["predictions_sha256"] = digest(output / "predictions.csv")
        return receipt.finish(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--representation-config", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_VARIATIONAL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.policy, args.representation_config, args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
