"""Download and validate the frozen research cohort before model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from quantum_neuro_ml.data.cohort import (
    DEFAULT_METADATA, DEFAULT_OUTPUT, DEFAULT_POLICY, CohortPolicy, QualitySubject,
    audit_metadata, cohort_counts, eligible_groups, write_audit,
)
from quantum_neuro_ml.data.connectivity import functional_connectivity, load_roi_timeseries
from quantum_neuro_ml.data.smoke import download_derivative
from quantum_neuro_ml.data.splits import acquisition_group
from quantum_neuro_ml.provenance import recorded_run


DEFAULT_RAW = Path("data/raw/cpac/filt_noglobal/rois_cc200")


def _build_dataset(
    *, metadata_path: Path = DEFAULT_METADATA, policy_path: Path = DEFAULT_POLICY,
    raw_dir: Path = DEFAULT_RAW, output_dir: Path = DEFAULT_OUTPUT,
    offline: bool = False, workers: int = 4,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if (output_dir / "report.json").exists() or (output_dir / "connectivity.npz").exists():
        raise ValueError("Research outputs already exist; use a new output directory")
    policy = CohortPolicy.load(policy_path)
    subjects, decisions, audit = audit_metadata(metadata_path, policy)
    if len({acquisition_group(s.subject.site_id) for s in subjects}) < 4:
        raise ValueError("Nested three-fold grouped evaluation requires at least four acquisition groups")
    write_audit(output_dir, subjects, decisions, audit)
    upper = np.triu_indices(200, k=1)

    def process(entry: QualitySubject) -> tuple[QualitySubject, np.ndarray | None, dict[str, object]]:
        # Download/cache errors stop the run; only invalid signals are exclusions.
        path, digest = download_derivative(entry.subject, raw_dir, offline=offline)
        record = {"subject_id": entry.subject.subject_id, "source_sha256": digest}
        try:
            timeseries = load_roi_timeseries(path)
            record["timepoints"] = timeseries.shape[0]
            if timeseries.shape[0] < policy.min_timepoints:
                return entry, None, dict(record, exclusion="scan_below_minimum_timepoints")
            _, fisher = functional_connectivity(timeseries)
        except ValueError as error:
            return entry, None, dict(record, exclusion="invalid_roi_signal", detail=str(error))
        return entry, fisher[upper], record

    valid, features, records, scan_exclusions = [], [], [], {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (entry, edges, record) in enumerate(executor.map(process, subjects), 1):
            records.append(record)
            if edges is None:
                scan_exclusions[entry.subject.subject_id] = record["exclusion"]
            else:
                valid.append(entry)
                features.append(edges)
            if index % 25 == 0 or index == len(subjects):
                print(f"Validated {index}/{len(subjects)} scans", flush=True)
    groups = eligible_groups(valid, policy)
    keep = [i for i, entry in enumerate(valid) if acquisition_group(entry.subject.site_id) in groups]
    final = tuple(valid[i] for i in keep)
    if len(groups) < 4:
        raise ValueError("Fewer than four acquisition groups remain after scan validation")
    for decision in decisions:
        if decision["reasons"]:
            continue
        if decision["subject_id"] in scan_exclusions:
            decision["reasons"].append(scan_exclusions[decision["subject_id"]])
        elif decision["acquisition_group"] not in groups:
            decision["reasons"].append("site_below_minimum_after_scan_validation")
    audit["scan_validation"] = "complete"
    audit["metadata_eligible"] = audit["included"]
    audit["included"] = cohort_counts(final)
    audit["first_exclusion_reason"] = dict(sorted(Counter(
        d["reasons"][0] for d in decisions if d["reasons"]
    ).items()))
    write_audit(output_dir, final, decisions, audit)
    dataset_path = output_dir / "connectivity.npz"
    temporary = dataset_path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, features=np.stack([features[i] for i in keep]),
            subject_ids=np.array([s.subject.subject_id for s in final]),
            site_ids=np.array([s.subject.site_id for s in final]),
            site_groups=np.array([acquisition_group(s.subject.site_id) for s in final]),
            labels=np.array([int(s.subject.diagnosis == 1) for s in final]),
            age=np.array([s.subject.age_at_scan for s in final]),
            sex=np.array([s.subject.sex for s in final]),
            mean_fd=np.array([s.mean_fd for s in final]),
        )
    temporary.replace(dataset_path)
    report = {
        "purpose": "research_dataset", "policy_sha256": policy.sha256,
        "policy": audit["policy"], "metadata_sha256": audit["metadata_sha256"],
        "manifest_sha256": audit["manifest_sha256"],
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "derivative": "cpac/filt_noglobal/rois_cc200", "feature_count": len(upper[0]),
        "included": cohort_counts(final), "scan_records": records,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    }
    print(f"Research dataset: {len(final)} participants, {len(groups)} acquisition groups", flush=True)
    return report


def build_dataset(
    *, metadata_path: Path = DEFAULT_METADATA, policy_path: Path = DEFAULT_POLICY,
    raw_dir: Path = DEFAULT_RAW, output_dir: Path = DEFAULT_OUTPUT,
    offline: bool = False, workers: int = 4,
) -> dict[str, object]:
    if (output_dir / "report.json").exists() or (output_dir / "connectivity.npz").exists():
        raise ValueError("Research outputs already exist; use a new output directory")
    configuration = {"experiment": "research-dataset-v1", "offline": offline, "workers": workers,
                     "derivative": "cpac/filt_noglobal/rois_cc200"}
    with recorded_run(output_dir, configuration, {"metadata": metadata_path, "policy": policy_path}) as run:
        report = _build_dataset(metadata_path=metadata_path, policy_path=policy_path, raw_dir=raw_dir,
                                output_dir=output_dir, offline=offline, workers=workers)
        return run.finish(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    build_dataset(metadata_path=args.metadata, policy_path=args.policy, raw_dir=args.raw_dir,
                  output_dir=args.output_dir, workers=args.workers, offline=args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
