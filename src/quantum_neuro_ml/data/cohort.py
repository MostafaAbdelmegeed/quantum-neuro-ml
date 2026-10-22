"""Apply a versioned ABIDE inclusion policy and write an aggregate audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from quantum_neuro_ml.data.abide import AbideSubject, build_derivative_url
from quantum_neuro_ml.data.splits import acquisition_group


QC_COLUMNS = (
    "qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2",
    "qc_anat_rater_3", "qc_func_rater_3",
)
DEFAULT_METADATA = Path("data/metadata/Phenotypic_V1_0b_preprocessed1.csv")
DEFAULT_POLICY = Path("config/cohort-v1.json")
DEFAULT_OUTPUT = Path("data/cohort/v1")


@dataclass(frozen=True)
class CohortPolicy:
    version: str
    metadata_sha256: str
    max_mean_fd: float
    min_timepoints: int
    min_site_subjects: int
    min_site_per_diagnosis: int
    required_qc: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> CohortPolicy:
        values = json.loads(path.read_text(encoding="utf-8"))
        values["required_qc"] = tuple(values["required_qc"])
        policy = cls(**values)
        if not math.isfinite(policy.max_mean_fd) or policy.max_mean_fd <= 0:
            raise ValueError("Motion threshold must be finite and positive")
        if policy.min_timepoints < 3 or policy.min_site_per_diagnosis < 1:
            raise ValueError("Invalid scan or diagnosis minimum")
        if policy.min_site_subjects < 2 or policy.required_qc != QC_COLUMNS:
            raise ValueError("Policy must specify all five manual QC columns")
        return policy

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class QualitySubject:
    subject: AbideSubject
    mean_fd: float


def _number(value: str | None) -> float | None:
    try:
        result = float(value or "")
    except ValueError:
        return None
    return result if math.isfinite(result) and result != -9999 else None


def eligible_groups(subjects: Sequence[QualitySubject], policy: CohortPolicy) -> set[str]:
    counts = Counter((acquisition_group(s.subject.site_id), s.subject.diagnosis) for s in subjects)
    return {
        group for group, _ in counts
        if counts[group, 1] >= policy.min_site_per_diagnosis
        and counts[group, 2] >= policy.min_site_per_diagnosis
        and counts[group, 1] + counts[group, 2] >= policy.min_site_subjects
    }


def cohort_counts(subjects: Sequence[QualitySubject]) -> dict[str, object]:
    per_group: dict[str, dict[str, int]] = {}
    for entry in subjects:
        subject = entry.subject
        group = acquisition_group(subject.site_id)
        counts = per_group.setdefault(group, {"autism": 0, "control": 0, "total": 0})
        counts[subject.diagnosis_name] += 1
        counts["total"] += 1
    def distribution(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
    return {
        "subjects": len(subjects),
        "site_labels": len({s.subject.site_id for s in subjects}),
        "acquisition_groups": len(per_group),
        "diagnoses": dict(sorted(Counter(s.subject.diagnosis_name for s in subjects).items())),
        "sex_codes": dict(sorted(Counter(str(s.subject.sex) for s in subjects).items())),
        "age_years": distribution([s.subject.age_at_scan for s in subjects]),
        "mean_fd_mm": distribution([s.mean_fd for s in subjects]),
        "by_acquisition_group": dict(sorted(per_group.items())),
    }


def audit_metadata(
    metadata_path: Path, policy: CohortPolicy,
) -> tuple[tuple[QualitySubject, ...], list[dict[str, object]], dict[str, object]]:
    payload = metadata_path.read_bytes()
    metadata_digest = hashlib.sha256(payload).hexdigest()
    if metadata_digest != policy.metadata_sha256:
        raise ValueError("Metadata fingerprint differs from the frozen policy")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"SUB_ID", "FILE_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "func_mean_fd", *policy.required_qc}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(reader.fieldnames or []))}")
    decisions, candidates = [], []
    seen_subjects, seen_files = set(), set()
    for row in reader:
        identifier = _number(row["SUB_ID"])
        if identifier is None or not identifier.is_integer() or identifier <= 0:
            raise ValueError("Missing or invalid participant ID")
        subject_id = int(identifier)
        if subject_id in seen_subjects:
            raise ValueError("Duplicate participant ID in source metadata")
        seen_subjects.add(subject_id)
        file_id, site = row["FILE_ID"].strip(), row["SITE_ID"].strip()
        if file_id and file_id != "no_filename":
            build_derivative_url(file_id)
            if file_id in seen_files:
                raise ValueError("Duplicate file ID in source metadata")
            seen_files.add(file_id)
        diagnosis, age, sex = (_number(row[k]) for k in ("DX_GROUP", "AGE_AT_SCAN", "SEX"))
        fd = _number(row["func_mean_fd"])
        reasons = []
        if not file_id or file_id == "no_filename":
            reasons.append("no_derivative")
        if not site or diagnosis not in (1, 2):
            reasons.append("invalid_site_or_diagnosis")
        if age is None or age <= 0 or sex not in (1, 2):
            reasons.append("missing_or_invalid_covariates")
        if fd is None or fd < 0:
            reasons.append("missing_or_invalid_motion")
        elif fd >= policy.max_mean_fd:
            reasons.append("motion_at_or_above_threshold")
        if any(row[column].strip().lower() != "ok" for column in policy.required_qc):
            reasons.append("manual_qc_not_all_ok")
        decisions.append({
            "subject_id": subject_id, "file_id": file_id, "site_id": site,
            "acquisition_group": acquisition_group(site), "reasons": reasons,
        })
        if not reasons:
            subject = AbideSubject(subject_id, site, file_id, int(diagnosis), age, int(sex))
            candidates.append(QualitySubject(subject, fd))
    candidates.sort(key=lambda s: (acquisition_group(s.subject.site_id), s.subject.subject_id))
    groups = eligible_groups(candidates, policy)
    included = tuple(s for s in candidates if acquisition_group(s.subject.site_id) in groups)
    for decision in decisions:
        if not decision["reasons"] and decision["acquisition_group"] not in groups:
            decision["reasons"].append("site_below_minimum")
    decisions.sort(key=lambda d: d["subject_id"])
    report = {
        "purpose": "metadata_cohort_audit", "policy": asdict(policy),
        "policy_sha256": policy.sha256, "metadata_sha256": metadata_digest,
        "source_rows": len(decisions),
        "first_exclusion_reason": dict(sorted(Counter(d["reasons"][0] for d in decisions if d["reasons"]).items())),
        "before_site_minimum": cohort_counts(candidates), "included": cohort_counts(included),
        "scan_validation": "pending",
    }
    return included, decisions, report


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_audit(
    output: Path, subjects: Sequence[QualitySubject], decisions: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(asdict(s.subject), acquisition_group=acquisition_group(s.subject.site_id),
                 mean_fd=s.mean_fd, derivative_url=build_derivative_url(s.subject.file_id)) for s in subjects]
    fields = ["subject_id", "site_id", "file_id", "diagnosis", "age_at_scan", "sex", "acquisition_group", "mean_fd", "derivative_url"]
    write_csv(output / "manifest.csv", rows, fields)
    write_csv(output / "decisions.csv", [dict(d, reasons=";".join(d["reasons"])) for d in decisions],
              ["subject_id", "file_id", "site_id", "acquisition_group", "reasons"])
    report["manifest_sha256"] = hashlib.sha256((output / "manifest.csv").read_bytes()).hexdigest()
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if (args.output_dir / "report.json").exists() or (args.output_dir / "connectivity.npz").exists():
        raise ValueError("Outputs contain a research dataset; use a new audit directory")
    policy = CohortPolicy.load(args.policy)
    subjects, decisions, report = audit_metadata(args.metadata, policy)
    write_audit(args.output_dir, subjects, decisions, report)
    counts = report["included"]
    print(f"Metadata eligible: {counts['subjects']} participants in {counts['acquisition_groups']} acquisition groups")
    print(f"Policy SHA256: {policy.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
