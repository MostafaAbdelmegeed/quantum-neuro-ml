"""Train matched MLP/quantum readouts of the verified, frozen graph encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from quantum_neuro_ml.baselines import load_research_dataset
from quantum_neuro_ml.data.cohort import DEFAULT_POLICY, CohortPolicy, write_csv
from quantum_neuro_ml.evaluation import EvaluationData, ModelPlan, evaluate_models
from quantum_neuro_ml.experiments import DEFAULT_DATASET, DEFAULT_EXPERIMENTS, validate_configuration
from quantum_neuro_ml.graph_experiment import DEFAULT_GRAPH, graph_plans, validate_graph_config
from quantum_neuro_ml.provenance import digest, recorded_run
from quantum_neuro_ml.variational import TrainableReadout
from quantum_neuro_ml.variational_experiment import DEFAULT_VARIATIONAL, paired_difference, validate_readout_config


DEFAULT_COMBINATION = Path("config/graph-variational-v6.json")


def trainable_plans(graph: dict, kernels: dict, readout: dict, cache: Path) -> list[ModelPlan]:
    templates = [p for p in graph_plans(graph, kernels, cache) if p.name.endswith("_logistic")]
    plans = []
    for template in templates:
        prefix = template.name.removesuffix("_logistic")
        for seed in readout["seeds"]:
            for family in ("mlp", "quantum"):
                model = TrainableReadout(family=family, seed=seed, epochs=readout["epochs"],
                                         batch_size=readout["batch_size"], weight_decay=readout["weight_decay"],
                                         depth=readout["quantum_depth"], hidden_units=readout["classical_hidden_units"])
                estimator = Pipeline([*template.estimator.steps[:-1], ("model", model)], memory=template.estimator.memory)
                plans.append(ModelPlan(f"{prefix}_seed{seed}_{family}", estimator,
                                       {"model__learning_rate": readout["learning_rates"]}))
    return plans


def align_reference(fold: dict, reference: dict) -> None:
    kind = fold["model"].split("_", 1)[0]
    name = f"{kind}_graph4_logistic"
    candidates = [f for f in reference["folds"] if f["held_out_group"] == fold["held_out_group"] and f["model"] == name]
    if len(candidates) != 1:
        raise ValueError("Reference lacks a unique graph representation for this fold")
    old = candidates[0]
    if fold["training_subject_ids_sha256"] != old["training_subject_ids_sha256"]:
        raise ValueError("Graph readouts received different training participants from v4")
    for key in ("training_representation_sha256", "test_representation_sha256"):
        if fold["diagnostics"][key] != old["diagnostics"][key]:
            raise ValueError("Frozen graph coordinates differ from the completed v4 reference")


def validate_reference(path: Path, dataset_sha256: str, graph: dict, kernels: dict) -> dict:
    reference = json.loads(path.read_text(encoding="utf-8"))
    receipt = json.loads(path.with_name("run.json").read_text(encoding="utf-8"))
    if (receipt["state"] != "complete" or digest(path) != receipt["report_sha256"]
            or digest(path.with_name("source.zip")) != receipt["source_snapshot_sha256"]):
        raise ValueError("Encoder reference does not have a valid completion receipt")
    if (reference["dataset_sha256"] != dataset_sha256 or reference["protocol_configuration"] != graph
            or reference["kernel_configuration"] != kernels):
        raise ValueError("Encoder reference dataset or configuration differs")
    import quantum_neuro_ml
    package = Path(quantum_neuro_ml.__file__).parent
    for name, expected in receipt["source_sha256"].items():
        if digest(package / name) != expected:
            raise ValueError(f"Encoder reference source differs from current source: {name}")
    for name, expected in receipt["environment_sha256"].items():
        if digest(package.parent.parent / name) != expected:
            raise ValueError(f"Encoder reference environment differs: {name}")
    return reference


def validate_combination(config: dict, graph: dict, readout: dict) -> None:
    expected = {
        "version": "abide-graph-variational-v6", "encoder_protocol": "abide-signed-graph-v4",
        "readout_protocol": "abide-variational-v3", "encoder_seed": 42, "readout_seeds": [42, 137, 2026],
        "representations": ["raw", "residual"], "dimensions": 4, "readouts": ["mlp", "quantum"],
        "encoder_training": "supervised_in_each_inner_and_outer_training_partition_then_frozen",
        "reference_checks": "source_dataset_configuration_and_actual_outer_representation_fingerprints",
        "reuse_encoder_cache": True, "joint_training": False,
    }
    if config != expected:
        raise ValueError("Unsupported graph/variational combination configuration")
    if (graph["version"] != config["encoder_protocol"] or graph["seed"] != config["encoder_seed"]
            or graph["embedding_dimensions"] != config["dimensions"] or graph["representations"] != config["representations"]
            or readout["version"] != config["readout_protocol"] or readout["seeds"] != config["readout_seeds"]
            or readout["dimensions"] != config["dimensions"] or readout["representations"] != config["representations"]):
        raise ValueError("Graph/variational protocols do not match the combined experiment")


def run(dataset: Path, policy_path: Path, kernel_path: Path, graph_path: Path, readout_path: Path,
        config_path: Path, reference_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment outputs already exist; choose a new directory")
    if not reference_path.is_file():
        raise ValueError("The v4 encoder reference must complete first")
    output.mkdir(parents=True, exist_ok=True)
    payload = Path(__file__).read_bytes()
    entrypoint = output / "entrypoint.json"
    entrypoint.write_text(json.dumps({"script_sha256": hashlib.sha256(payload).hexdigest(),
                                      "script_utf8": payload.decode("utf-8")}, sort_keys=True), encoding="utf-8")
    inputs = {"dataset": dataset, "dataset_report": dataset.with_name("report.json"),
              "manifest": dataset.with_name("manifest.csv"), "policy": policy_path,
              "kernel_protocol": kernel_path, "graph_protocol": graph_path, "readout_protocol": readout_path,
              "combined_protocol": config_path, "encoder_reference": reference_path,
              "encoder_receipt": reference_path.with_name("run.json"), "runner_script": Path(__file__),
              "entrypoint_snapshot": entrypoint}
    with recorded_run(output, {"experiment": "frozen-graph-variational-v6"}, inputs) as receipt:
        if digest(Path(__file__)) != hashlib.sha256(payload).hexdigest():
            raise ValueError("Entrypoint source changed while being captured")
        config = json.loads(config_path.read_text())
        graph = json.loads(graph_path.read_text())
        kernels = json.loads(kernel_path.read_text())
        readout = json.loads(readout_path.read_text())
        policy = CohortPolicy.load(policy_path)
        validate_configuration(kernels, policy)
        validate_graph_config(graph)
        validate_readout_config(readout)
        validate_combination(config, graph, readout)
        arrays, lineage = load_research_dataset(dataset, policy)
        reference = validate_reference(reference_path, lineage["dataset_sha256"], graph, kernels)
        features = np.column_stack([arrays["age"], arrays["sex"], arrays["mean_fd"], arrays["features"]])
        data = EvaluationData(features, arrays["labels"], arrays["site_groups"], arrays["subject_ids"])
        plans = trainable_plans(graph, kernels, readout, reference_path.parent / "cache")
        def progress(fold, rows):
            align_reference(fold, reference)
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"fold": fold, "predictions": rows}, sort_keys=True) + "\n")
        report, predictions = evaluate_models(data, plans, seed=kernels["seed"], on_fold=progress)
        report.update(experiment_version=config["version"], protocol_configuration=config,
                      graph_configuration=graph, readout_configuration=readout, kernel_configuration=kernels,
                      encoder_reference_report_sha256=digest(reference_path),
                      entrypoint_sha256=hashlib.sha256(payload).hexdigest(),
                      dataset_sha256=lineage["dataset_sha256"], policy_sha256=lineage["policy_sha256"],
                      manifest_sha256=lineage["manifest_sha256"], metadata_sha256=lineage["metadata_sha256"])
        report["paired_comparisons"] = {}
        for kind in config["representations"]:
            for seed in config["readout_seeds"]:
                prefix = f"{kind}_graph4_seed{seed}"
                report["paired_comparisons"][prefix + "_quantum_minus_mlp"] = paired_difference(
                    report, prefix + "_quantum", prefix + "_mlp", kernels["seed"])
        write_csv(output / "predictions.csv", predictions,
                  ["model", "subject_id", "held_out_group", "label", "prediction", "score", "probability"])
        report["predictions_sha256"] = digest(output / "predictions.csv")
        return receipt.finish(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--kernel-config", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--graph-config", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--readout-config", type=Path, default=DEFAULT_VARIATIONAL)
    parser.add_argument("--config", type=Path, default=DEFAULT_COMBINATION)
    parser.add_argument("--encoder-reference", type=Path, default=Path("results/graph-v4-bounded/report.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.policy, args.kernel_config, args.graph_config, args.readout_config,
        args.config, args.encoder_reference, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
