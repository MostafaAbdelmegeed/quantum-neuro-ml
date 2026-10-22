"""Download a small ABIDE cohort and verify connectivity construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from quantum_neuro_ml.data.abide import (
    PHENOTYPIC_URL,
    AbideSubject,
    build_derivative_url,
    fetch_phenotypic_csv,
    parse_phenotypic_csv,
    write_manifest,
)
from quantum_neuro_ml.data.connectivity import (
    FISHER_CLIP,
    functional_connectivity,
    load_roi_timeseries,
)
from quantum_neuro_ml.data.splits import leave_one_site_out


def select_smoke_cohort(
    subjects: Sequence[AbideSubject],
    sites: Sequence[str],
    per_group: int = 2,
) -> tuple[AbideSubject, ...]:
    """Choose the lowest subject IDs per site and diagnosis without replacement."""

    if len(sites) < 2 or len(set(sites)) != len(sites):
        raise ValueError("Choose at least two distinct sites")
    if per_group < 1:
        raise ValueError("per_group must be positive")
    if len({s.subject_id for s in subjects}) != len(subjects):
        raise ValueError("Duplicate participant IDs in source cohort")
    if len({s.file_id for s in subjects}) != len(subjects):
        raise ValueError("Duplicate file IDs in source cohort")

    selected: list[AbideSubject] = []
    for site in sorted(sites):
        for diagnosis in (1, 2):
            candidates = sorted(
                (s for s in subjects if s.site_id == site and s.diagnosis == diagnosis),
                key=lambda s: s.subject_id,
            )
            if len(candidates) < per_group:
                raise ValueError(f"Site {site}, diagnosis {diagnosis}: insufficient subjects")
            selected.extend(candidates[:per_group])
    return tuple(selected)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_derivative(
    subject: AbideSubject,
    raw_dir: Path,
    *,
    offline: bool = False,
    timeout: float = 30.0,
) -> tuple[Path, str]:
    """Cache a derivative and verify its SHA-256 receipt on every reuse.

    The receipt detects local changes after download; it is not a checksum
    supplied by the dataset publisher. Incomplete caches require explicit repair.
    """

    url = build_derivative_url(subject.file_id)
    path = raw_dir / f"{subject.file_id}_rois_cc200.1D"
    receipt = path.with_name(path.name + ".json")
    if path.exists() or receipt.exists():
        if not path.is_file() or not receipt.is_file():
            raise ValueError(f"Incomplete download cache: {path}")
        record = json.loads(receipt.read_text(encoding="utf-8"))
        digest = _digest(path)
        if not isinstance(record, dict) or record.get("url") != url or record.get("sha256") != digest:
            raise ValueError(f"Download checksum or URL mismatch: {path}")
        return path, digest
    if offline:
        raise ValueError(f"Offline mode requires a cached derivative: {path}")

    request = urllib.request.Request(url, headers={"User-Agent": "quantum-neuro-ml/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Derivative request returned {response.status}: {url}")
        expected_bytes = response.headers.get("Content-Length")
        payload = response.read(10_000_001)
    if not payload or len(payload) > 10_000_000:
        raise ValueError(f"Empty or oversized ROI derivative: {url}")
    if expected_bytes is not None and len(payload) != int(expected_bytes):
        raise ValueError(f"Incomplete ROI derivative response: {url}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    digest = _digest(path)
    _write_json(receipt, {"url": url, "sha256": digest, "bytes": len(payload)})
    return path, digest


def run_smoke(
    *,
    metadata_path: Path,
    raw_dir: Path,
    output_dir: Path,
    sites: Sequence[str] = ("NYU", "PITT"),
    held_out_site: str = "PITT",
    per_group: int = 2,
    offline: bool = False,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Build a local, balanced integration cohort; do not fit any model or transform."""

    if metadata_path.exists():
        metadata_text = metadata_path.read_text(encoding="utf-8-sig")
    elif offline:
        raise ValueError(f"Offline mode requires cached metadata: {metadata_path}")
    else:
        metadata_text = fetch_phenotypic_csv(timeout)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_bytes(metadata_text.encode("utf-8"))

    subjects = select_smoke_cohort(parse_phenotypic_csv(metadata_text), sites, per_group)
    split = leave_one_site_out(subjects, held_out_site)
    test_ids = {s.subject_id for s in split.test}
    correlations, fisher_matrices, records = [], [], []
    for subject in subjects:
        path, digest = download_derivative(subject, raw_dir, offline=offline, timeout=timeout)
        timeseries = load_roi_timeseries(path)
        correlation, fisher_z = functional_connectivity(timeseries)
        correlations.append(correlation)
        fisher_matrices.append(fisher_z)
        records.append({
            "subject_id": subject.subject_id,
            "site": subject.site_id,
            "diagnosis": subject.diagnosis_name,
            "partition": "test" if subject.subject_id in test_ids else "train",
            "timepoints": timeseries.shape[0],
            "rois": timeseries.shape[1],
            "source_sha256": digest,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_digest = write_manifest(subjects, output_dir / "cohort.csv")
    correlation_array = np.stack(correlations)
    fisher_array = np.stack(fisher_matrices)
    upper_triangle = np.triu_indices(200, k=1)
    dataset_path = output_dir / "connectivity.npz"
    temporary = dataset_path.with_name(dataset_path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            correlation=correlation_array,
            fisher_z=fisher_array,
            features=fisher_array[:, upper_triangle[0], upper_triangle[1]],
            subject_ids=np.array([s.subject_id for s in subjects]),
            site_ids=np.array([s.site_id for s in subjects]),
            labels=np.array([int(s.diagnosis == 1) for s in subjects]),
            is_test=np.array([s.subject_id in test_ids for s in subjects]),
            roi_labels=np.arange(1, 201),
        )
    temporary.replace(dataset_path)

    report = {
        "purpose": "integration_smoke_test",
        "metadata_url": PHENOTYPIC_URL,
        "metadata_sha256": _digest(metadata_path),
        "manifest_sha256": manifest_digest,
        "dataset_sha256": _digest(dataset_path),
        "derivative": "cpac/filt_noglobal/rois_cc200",
        "selection": "lowest subject IDs per site and diagnosis",
        "per_group": per_group,
        "held_out_site": held_out_site,
        "subjects": len(subjects),
        "sites": sorted(sites),
        "train_subjects": len(split.train),
        "test_subjects": len(split.test),
        "diagnoses": dict(sorted(Counter(s.diagnosis_name for s in subjects).items())),
        "feature_count": len(upper_triangle[0]),
        "fisher_clip": FISHER_CLIP,
        "numpy_version": np.__version__,
        "records": records,
    }
    _write_json(output_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(
        "data/metadata/Phenotypic_V1_0b_preprocessed1.csv"
    ))
    parser.add_argument("--raw-dir", type=Path, default=Path(
        "data/raw/cpac/filt_noglobal/rois_cc200"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path("data/smoke"))
    parser.add_argument("--sites", nargs="+", default=["NYU", "PITT"])
    parser.add_argument("--held-out-site", default="PITT")
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        report = run_smoke(
            metadata_path=args.metadata,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            sites=args.sites,
            held_out_site=args.held_out_site,
            per_group=args.per_group,
            offline=args.offline,
            timeout=args.timeout,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Verified {report['subjects']} subjects across {len(report['sites'])} sites; "
        f"train={report['train_subjects']}, test={report['test_subjects']}; "
        f"{report['feature_count']} edges per subject."
    )
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
