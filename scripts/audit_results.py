"""Recompute saved metrics without sklearn and verify completed-run receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from quantum_neuro_ml.provenance import digest, recorded_run


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equal(first: float, second: float, message: str) -> None:
    require(bool(np.isclose(first, second, atol=1e-12, rtol=0)), message)


def independent_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict:
    require(set(labels) == {0, 1} and set(predictions) <= {0, 1}, "Invalid binary outcomes")
    positive, negative = labels == 1, labels == 0
    sensitivity, specificity = predictions[positive].mean(), (1 - predictions[negative]).mean()
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n1, n0 = positive.sum(), negative.sum()
    auc = (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"balanced_accuracy": float((sensitivity + specificity) / 2), "auroc": float(auc)}


def verify_receipt(directory: Path) -> dict:
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    require(run["state"] == "complete", f"Run is incomplete: {directory}")
    require(digest(directory / "report.json") == run["report_sha256"], "Completion report digest differs")
    require(digest(directory / "source.zip") == run["source_snapshot_sha256"], "Source archive digest differs")
    with zipfile.ZipFile(directory / "source.zip") as archive:
        encoded = archive.read("experiment.json")
        require(hashlib.sha256(encoded).hexdigest() == run["experiment_sha256"], "Archived experiment digest differs")
        require(json.loads(encoded) == run["configuration"], "Archived experiment configuration differs")
        producer = archive.read("src/quantum_neuro_ml/provenance.py").decode("utf-8")
        archives_inputs = 'archive.writestr(f"inputs/{name}/{path.name}", self.input_payloads[name])' in producer
        for name, expected in run["source_sha256"].items():
            payload = archive.read(f"src/quantum_neuro_ml/{name}")
            require(hashlib.sha256(payload).hexdigest() == expected, f"Archived source differs: {name}")
        for name, expected in run["environment_sha256"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, f"Archived environment differs: {name}")
        for name, record in run["inputs"].items():
            path = Path(record["path"])
            if name == "runner_script" and "entrypoint_snapshot" in run["inputs"]:
                snapshot_record = run["inputs"]["entrypoint_snapshot"]
                snapshot_path = Path(snapshot_record["path"])
                snapshot_bytes = (archive.read(f"inputs/entrypoint_snapshot/{snapshot_path.name}")
                                 if archives_inputs else snapshot_path.read_bytes())
                require(hashlib.sha256(snapshot_bytes).hexdigest() == snapshot_record["sha256"],
                        "Entrypoint snapshot differs from its receipt")
                snapshot = json.loads(snapshot_bytes)
                script_bytes = snapshot["script_utf8"].encode("utf-8")
                require(hashlib.sha256(script_bytes).hexdigest() == record["sha256"]
                        and snapshot["script_sha256"] == record["sha256"],
                        "Archived runner script differs from its input receipt")
                continue
            require(digest(path) == record["sha256"], f"Recorded input differs: {name}")
            if archives_inputs and path.suffix.lower() in (".json", ".csv"):
                archived = archive.read(f"inputs/{name}/{path.name}")
                require(hashlib.sha256(archived).hexdigest() == record["sha256"], f"Archived input differs: {name}")
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    for key in ("source_sha256", "source_snapshot_sha256", "experiment_sha256"):
        require(report[key] == run[key], f"Report and receipt differ: {key}")
    if "runner_script" in run["inputs"]:
        recorded_script = run["inputs"]["runner_script"]["sha256"]
        require(report.get("entrypoint_sha256", report.get("script_sha256")) == recorded_script,
                "Runner entrypoint digest differs from its archived source")
    if "encoder_cache_manifest" in run["inputs"]:
        manifest = run["inputs"]["encoder_cache_manifest"]
        require(report.get("encoder_cache_manifest_sha256") == manifest["sha256"],
                "Encoder cache manifest digest differs from the report")
        require(report.get("encoder_cache_origin") == str(Path(run["inputs"]["encoder_cache_origin_receipt"]["path"]).parent),
                "Encoder cache origin differs from its receipt")
        for relative, expected in json.loads(Path(manifest["path"]).read_text(encoding="utf-8")).items():
            item = Path(relative)
            require(not item.is_absolute() and ".." not in item.parts, "Unsafe path in encoder cache manifest")
            require(digest(directory / "cache" / item) == expected, "Reused encoder cache entry changed")
    return report


def training_indices(ids, labels, groups, held_out, fraction, seed):
    candidates = np.flatnonzero(groups != held_out)
    if fraction == 1:
        return candidates
    selected = []
    for group in sorted(set(groups[candidates])):
        for label in (0, 1):
            stratum = candidates[(groups[candidates] == group) & (labels[candidates] == label)]
            stratum = stratum[np.argsort(ids[stratum])]
            count = min(len(stratum), max(2, int(np.ceil(len(stratum) * fraction))))
            token = hashlib.sha256(f"{seed}:{group}:{label}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
            selected.extend(rng.permutation(stratum)[:count].tolist())
    return np.array(sorted(selected), dtype=int)


def audit_predictions(directory: Path, ids: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict:
    report = verify_receipt(directory)
    require(digest(directory / "predictions.csv") == report["predictions_sha256"], "Prediction digest differs")
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {int(subject): (int(label), str(group)) for subject, label, group in zip(ids, labels, groups)}
    models = sorted({r["model"] for r in rows})
    require(set(models) == set(report["summaries"]), "Model inventory differs")
    require(len(rows) == len(ids) * len(models), "Prediction count differs")
    totals = {}
    folds = {(f["model"], f["held_out_group"]): f for f in report["folds"]}
    require(len(report["folds"]) == len(folds) == len(models) * len(set(groups)), "Fold inventory differs or repeats")
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        require(len({r["subject_id"] for r in model_rows}) == len(ids), "A participant was predicted twice or omitted")
        convergence_warnings, capped_refits = 0, 0
        for row in model_rows:
            subject = int(row["subject_id"])
            require(subject in expected and (int(row["label"]), row["held_out_group"]) == expected[subject],
                    "Prediction identity, diagnosis, or group differs from the dataset")
            require(np.isfinite(float(row["score"])), "Nonfinite decision score")
            if row["probability"]:
                require(0 <= float(row["probability"]) <= 1, "Invalid probability")
        group_metrics = []
        for group in sorted(set(groups)):
            subset = [r for r in model_rows if r["held_out_group"] == group]
            values = independent_metrics(np.array([int(r["label"]) for r in subset]),
                                         np.array([int(r["prediction"]) for r in subset]),
                                         np.array([float(r["score"]) for r in subset]))
            fold = folds[(model, group)]
            diagnostics = fold.get("diagnostics", {})
            warning_count = diagnostics.get("convergence_warning_count", 0)
            require(isinstance(warning_count, int) and warning_count >= 0,
                    "Invalid convergence-warning count")
            convergence_warnings += warning_count
            if "readout_fit_status" in diagnostics:
                status = diagnostics["readout_fit_status"]
                require(status in (0, 1), "Unknown readout solver status")
                if status == 1:
                    require(warning_count > 0, "Capped readout lacks a convergence warning")
                    capped_refits += 1
            if "experiment_version" in report:
                require("training_subject_ids_sha256" in fold, "Follow-up fold lacks its training identity receipt")
            require(len(subset) == fold["test_subjects"] == int(np.sum(groups == group)), "Held-out group count differs")
            for metric in values:
                equal(values[metric], fold["metrics"][metric], f"Saved fold metric differs: {model}/{group}/{metric}")
            if "training_subject_ids_sha256" in fold:
                train = training_indices(ids, labels, groups, group, report.get("training_fraction", 1), report["seed"])
                values_to_hash = np.ascontiguousarray(ids[train])
                fingerprint = hashlib.sha256(str((values_to_hash.shape, values_to_hash.dtype.str)).encode()
                                             + values_to_hash.tobytes()).hexdigest()
                require(fingerprint == fold["training_subject_ids_sha256"], "Training participant fingerprint differs")
                require(len(train) == fold["train_subjects"], "Training size differs")
                require(set(fold["training_groups"]) == set(groups[train]), "Training group inventory differs")
                validation_groups = []
                for inner in fold["inner_folds"]:
                    a, b = set(inner["train_groups"]), set(inner["validation_groups"])
                    require(not a & b and group not in a | b and a | b == set(groups[train]), "Inner group leakage")
                    validation_groups.extend(b)
                require(sorted(validation_groups) == sorted(set(groups[train])), "Inner validation groups repeat or are missing")
            group_metrics.append(values)
        totals[model] = {}
        for metric in ("balanced_accuracy", "auroc"):
            values = np.array([v[metric] for v in group_metrics])
            mean = float(values.mean())
            equal(mean, report["summaries"][model]["site_mean"][metric], "Saved site-mean metric differs")
            rng = np.random.default_rng(report["seed"])
            bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
            interval = np.quantile(bootstrap, [0.025, 0.975])
            np.testing.assert_allclose(interval, report["summaries"][model]["site_bootstrap_95_percent"][metric], atol=1e-12, rtol=0)
            totals[model][metric] = mean
        pooled = independent_metrics(np.array([int(r["label"]) for r in model_rows]),
                                     np.array([int(r["prediction"]) for r in model_rows]),
                                     np.array([float(r["score"]) for r in model_rows]))
        for metric in pooled:
            equal(pooled[metric], report["summaries"][model]["pooled_out_of_fold"][metric], "Saved pooled metric differs")
        totals[model]["convergence_warning_count"] = convergence_warnings
        totals[model]["capped_refit_count"] = capped_refits
    # Compare actual input bytes rather than assuming identical pipeline names.
    matched = 0
    for model in models:
        if not model.endswith("_quantum"):
            continue
        prefix = model.removesuffix("_quantum")
        for family in ("mlp", "logistic", "rbf", "polynomial"):
            other = prefix + "_" + family
            if other not in models:
                continue
            for group in set(groups):
                a, b = folds[(model, group)], folds[(other, group)]
                require(a["training_subject_ids_sha256"] == b["training_subject_ids_sha256"], "Matched training IDs differ")
                for key in ("training_representation_sha256", "test_representation_sha256"):
                    require(a["diagnostics"][key] == b["diagnostics"][key], "Matched representation differs")
                matched += 1
    return {"directory": str(directory), "models": len(models), "predictions": len(rows),
            "folds": len(folds), "matched_readout_fold_pairs": matched,
            "report_sha256": digest(directory / "report.json"), "verified_site_means": totals}


def audit_permutations(directory: Path, ids: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict:
    report = verify_receipt(directory)
    run = json.loads((directory / "run.json").read_text())
    reference = Path(run["inputs"]["reference_report"]["path"])
    reference_report = json.loads(reference.read_text())
    equal(report["observed_balanced_accuracy"], reference_report["summaries"]["logistic"]["site_mean"]["balanced_accuracy"],
          "Permutation observed statistic differs from the recorded reference")
    with Path(run["inputs"]["reference_predictions"]["path"]).open(newline="") as stream:
        original_rows = {int(r["subject_id"]): r for r in csv.DictReader(stream) if r["model"] == "logistic"}
    with (directory / "observed-predictions.csv").open(newline="") as stream:
        observed_rows = list(csv.DictReader(stream))
    require(len(observed_rows) == len(original_rows) == len(ids), "Observed permutation prediction count differs")
    require(len({r["subject_id"] for r in observed_rows}) == len(ids), "Observed permutation predictions repeat")
    expected = {int(subject): (int(label), str(group)) for subject, label, group in zip(ids, labels, groups)}
    for row in observed_rows:
        subject = int(row["subject_id"])
        require(subject in expected and (int(row["label"]), row["held_out_group"]) == expected[subject],
                "Observed permutation prediction identity differs")
        old = original_rows[subject]
        require(row["model"] == "logistic" and int(row["prediction"]) == int(old["prediction"]),
                "Observed permutation predictions differ from the recorded reference")
        equal(float(row["score"]), float(old["score"]), "Observed permutation scores differ from reference")
    observed_metrics = []
    observed_folds = report["observed"]["folds"]
    require(len(observed_folds) == len(set(groups)) and len({f["held_out_group"] for f in observed_folds}) == len(set(groups)),
            "Observed permutation folds repeat or are missing")
    for group in sorted(set(groups)):
        rows = [r for r in observed_rows if r["held_out_group"] == group]
        values = independent_metrics(np.array([int(r["label"]) for r in rows]),
                                     np.array([int(r["prediction"]) for r in rows]),
                                     np.array([float(r["score"]) for r in rows]))
        fold = next(f for f in observed_folds if f["held_out_group"] == group)
        for metric in values:
            equal(values[metric], fold["metrics"][metric], "Observed permutation fold metric differs")
        observed_metrics.append(values)
    observed = float(np.mean([m["balanced_accuracy"] for m in observed_metrics]))
    equal(observed, report["observed_balanced_accuracy"], "Observed permutation statistic differs from predictions")
    for metric in ("balanced_accuracy", "auroc"):
        mean = float(np.mean([m[metric] for m in observed_metrics]))
        equal(mean, report["observed"]["summaries"]["logistic"]["site_mean"][metric],
              "Nested observed summary differs from predictions")
    trials = report["trials"]
    require(len(trials) == report["permutations"] == 99, "Permutation count differs from frozen protocol")
    require(sorted(t["trial"] for t in trials) == list(range(99)), "Permutation trials repeat or are missing")
    hashes = []
    for trial in trials:
        rng = np.random.default_rng(np.random.SeedSequence([report["seed"], trial["trial"] + 1]))
        permuted = labels.copy()
        for group in sorted(set(groups)):
            indices = np.flatnonzero(groups == group)
            permuted[indices] = labels[rng.permutation(indices)]
            require(np.bincount(permuted[indices], minlength=2).tolist() == np.bincount(labels[indices], minlength=2).tolist(),
                    "Within-group class counts changed")
        fingerprint = hashlib.sha256(permuted.tobytes()).hexdigest()
        require(fingerprint == trial["permuted_labels_sha256"], "Permutation labels differ from seed")
        hashes.append(fingerprint)
        require(set(trial["by_group"]) == set(groups), "Permutation group inventory differs")
        for metric in ("balanced_accuracy", "auroc"):
            require(all(0 <= m[metric] <= 1 for m in trial["by_group"].values()), "Invalid permutation metric")
            mean = np.mean([m[metric] for m in trial["by_group"].values()])
            equal(mean, trial[metric], "Permutation site mean differs")
    require(len(set(hashes)) == 99, "A shuffled target vector repeats")
    scores = np.array([r["balanced_accuracy"] for r in trials])
    exceedances = int(np.sum(scores >= report["observed_balanced_accuracy"] - 1e-12))
    require(exceedances == report["exceedances"], "Permutation exceedance count differs")
    equal((1 + exceedances) / 100, report["empirical_one_sided_p"], "Permutation p-value differs")
    equal(float(scores.mean()), report["null_mean"], "Null mean differs")
    np.testing.assert_allclose(np.quantile(scores, [0.025, 0.5, 0.975]), report["null_percentiles"], atol=1e-12, rtol=0)
    return {"directory": str(directory), "permutations": 99, "unique_target_vectors": 99,
            "observed_balanced_accuracy": report["observed_balanced_accuracy"],
            "null_mean": report["null_mean"], "empirical_one_sided_p": report["empirical_one_sided_p"],
            "report_sha256": digest(directory / "report.json"),
            "scope": "Observed predictions recomputed; null target construction and statistic arithmetic checked; null model predictions not independently refitted"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/cohort/v1-verified/connectivity.npz"))
    parser.add_argument("--result-dir", type=Path, action="append", required=True)
    parser.add_argument("--permutation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {"dataset": args.dataset, "audit_script": Path(__file__)}
    for index, directory in enumerate(args.result_dir):
        inputs[f"report_{index}"] = directory / "report.json"
        inputs[f"predictions_{index}"] = directory / "predictions.csv"
        inputs[f"receipt_{index}"] = directory / "run.json"
    if args.permutation_dir:
        inputs["permutation_report"] = args.permutation_dir / "report.json"
        inputs["permutation_receipt"] = args.permutation_dir / "run.json"
    with recorded_run(args.output_dir, {"experiment": "independent-artifact-audit"}, inputs) as receipt:
        with np.load(args.dataset, allow_pickle=False) as arrays:
            ids, labels, groups = arrays["subject_ids"], arrays["labels"], arrays["site_groups"]
        shutil.copy2(__file__, args.output_dir / "audit_script.py")
        results = [audit_predictions(directory, ids, labels, groups) for directory in args.result_dir]
        report = {"audit": "passed", "subjects": len(ids), "acquisition_groups": len(set(groups)),
                  "script_sha256": digest(Path(__file__)), "experiments": results}
        if args.permutation_dir:
            report["permutation_test"] = audit_permutations(args.permutation_dir, ids, labels, groups)
        receipt.finish(report)
        print(json.dumps({"audit": "passed", "experiments": len(results),
                          "models": sum(r["models"] for r in results),
                          "predictions": sum(r["predictions"] for r in results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
