"""Measure outer-fold stability of the learned co-variation edge modules."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from quantum_neuro_ml.baselines import load_research_dataset
from quantum_neuro_ml.covariation import EdgeCovariationGroups, compare_modules
from quantum_neuro_ml.data.cohort import DEFAULT_POLICY, CohortPolicy
from quantum_neuro_ml.experiments import DEFAULT_DATASET
from quantum_neuro_ml.provenance import digest, recorded_run
from quantum_neuro_ml.representations import NuisanceResidualizer


def run(dataset: Path, policy_path: Path, protocol_path: Path,
        experiment_dir: Path, output: Path) -> dict:
    run_record = json.loads((experiment_dir / "run.json").read_text(encoding="utf-8"))
    if run_record.get("state") != "complete" or digest(experiment_dir / "report.json") != run_record.get("report_sha256"):
        raise ValueError("Co-variation experiment receipt is incomplete or invalid")
    inputs = {"dataset": dataset, "policy": policy_path, "protocol": protocol_path,
              "experiment_receipt": experiment_dir / "run.json",
              "experiment_report": experiment_dir / "report.json"}
    with recorded_run(output, {"experiment": "edge-covariation-module-stability-v1"}, inputs) as receipt:
        policy = CohortPolicy.load(policy_path)
        config = json.loads(protocol_path.read_text(encoding="utf-8"))
        arrays, lineage = load_research_dataset(dataset, policy)
        values = np.column_stack([arrays["age"], arrays["sex"], arrays["mean_fd"], arrays["features"]])
        groups = np.asarray(arrays["site_groups"])
        fitted = {}
        for kind in config["representations"]:
            for dimensions in config["dimensions"]:
                key = f"{kind}_covar{dimensions}"
                fitted[key] = {}
                for held_out in sorted(set(groups)):
                    train = groups != held_out
                    if kind == "raw":
                        training_edges = values[train, 3:]
                    else:
                        residualizer = NuisanceResidualizer().fit(values[train])
                        training_edges = residualizer.transform(values[train])
                    transformer = EdgeCovariationGroups(
                        dimensions=dimensions, candidate_edges=config["candidate_edges"]
                    ).fit(training_edges)
                    modules = []
                    for edge_ids, signs in zip(transformer.modules_, transformer.signs_):
                        modules.append({int(edge): int(sign) for edge, sign in zip(edge_ids, signs)})
                    fitted[key][str(held_out)] = {
                        "candidate_edges": set(map(int, transformer.candidate_indices_)),
                        "modules": modules,
                        "diagnostics": transformer.diagnostics_,
                    }

        summaries = {}
        for key, sites in fitted.items():
            candidate_overlap, module_similarity = [], []
            for first_site, second_site in combinations(sorted(sites), 2):
                first, second = sites[first_site], sites[second_site]
                union = first["candidate_edges"] | second["candidate_edges"]
                candidate_overlap.append(len(first["candidate_edges"] & second["candidate_edges"]) / len(union))
                similarity, _ = compare_modules(first["modules"], second["modules"])
                module_similarity.append(similarity)
            summaries[key] = {
                "outer_site_pairs": len(candidate_overlap),
                "mean_candidate_edge_jaccard": float(np.mean(candidate_overlap)),
                "mean_signed_module_jaccard_after_best_matching": float(np.mean(module_similarity)),
                "site_fits": {site: value["diagnostics"] for site, value in sites.items()},
            }
        report = {"experiment_version": "edge-covariation-module-stability-v1",
                  "dataset_sha256": lineage["dataset_sha256"], "policy_sha256": lineage["policy_sha256"],
                  "covariation_report_sha256": digest(experiment_dir / "report.json"),
                  "interpretation": "Outer-training-fold overlap diagnostic; no diagnosis labels used.",
                  "summaries": summaries}
        return receipt.finish(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--protocol", type=Path, default=Path("config/covariation-v1.json"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("results/covariation-v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.policy, args.protocol, args.experiment_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
