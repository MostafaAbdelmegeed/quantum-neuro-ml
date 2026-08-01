"""Build reproducible manifests for ABIDE Preprocessed derivatives.

The module intentionally uses only the Python standard library. Heavy numerical
dependencies will be introduced after the data contract and split logic are stable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ABIDE_ROOT = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENOTYPIC_URL = f"{ABIDE_ROOT}/Phenotypic_V1_0b_preprocessed1.csv"

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
_VALID_PIPELINES = frozenset({"ccs", "cpac", "dparsf", "niak"})
_VALID_STRATEGIES = frozenset(
    {"filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"}
)


@dataclass(frozen=True, slots=True)
class AbideDerivative:
    """A single PCP preprocessing and derivative configuration."""

    pipeline: str = "cpac"
    strategy: str = "filt_noglobal"
    derivative: str = "rois_cc200"
    extension: str = "1D"

    def __post_init__(self) -> None:
        if self.pipeline not in _VALID_PIPELINES:
            raise ValueError(f"Unsupported pipeline: {self.pipeline}")
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        for name, value in (
            ("derivative", self.derivative),
            ("extension", self.extension),
        ):
            if not _SAFE_IDENTIFIER.fullmatch(value):
                raise ValueError(f"Unsafe {name}: {value!r}")


@dataclass(frozen=True, slots=True)
class AbideSubject:
    """Minimal subject metadata needed for a reproducible download manifest."""

    subject_id: int
    site_id: str
    file_id: str
    diagnosis: int
    age_at_scan: float | None
    sex: int | None

    @property
    def diagnosis_name(self) -> str:
        return {1: "autism", 2: "control"}.get(self.diagnosis, "unknown")


def build_derivative_url(
    file_id: str,
    derivative: AbideDerivative = AbideDerivative(),
) -> str:
    """Return the public S3 URL for one participant derivative."""

    if not _SAFE_IDENTIFIER.fullmatch(file_id):
        raise ValueError(f"Unsafe ABIDE file identifier: {file_id!r}")
    return (
        f"{ABIDE_ROOT}/Outputs/{derivative.pipeline}/{derivative.strategy}/"
        f"{derivative.derivative}/{file_id}_{derivative.derivative}."
        f"{derivative.extension}"
    )


def fetch_phenotypic_csv(timeout_seconds: float = 30.0) -> str:
    """Fetch the public PCP phenotypic summary as UTF-8 text."""

    request = urllib.request.Request(
        PHENOTYPIC_URL,
        headers={"User-Agent": "quantum-neuro-ml/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise RuntimeError(f"ABIDE metadata request returned {response.status}")
        return response.read().decode("utf-8-sig")


def parse_phenotypic_csv(csv_text: str) -> list[AbideSubject]:
    """Parse subjects with downloadable derivative identifiers.

    Rows marked ``no_filename`` are excluded because PCP does not expose a
    derivative file for them. Diagnosis values are retained as published:
    1 is autism spectrum disorder and 2 is typical control.
    """

    rows = csv.DictReader(io.StringIO(csv_text))
    subjects: list[AbideSubject] = []
    for row in rows:
        file_id = (row.get("FILE_ID") or "").strip()
        if not file_id or file_id == "no_filename":
            continue

        diagnosis = _optional_int(row.get("DX_GROUP"))
        subject_id = _optional_int(row.get("SUB_ID"))
        site_id = (row.get("SITE_ID") or "").strip()
        if subject_id is None or diagnosis not in {1, 2} or not site_id:
            continue

        subjects.append(
            AbideSubject(
                subject_id=subject_id,
                site_id=site_id,
                file_id=file_id,
                diagnosis=diagnosis,
                age_at_scan=_optional_float(row.get("AGE_AT_SCAN")),
                sex=_optional_int(row.get("SEX")),
            )
        )

    _validate_unique_file_ids(subjects)
    return sorted(subjects, key=lambda subject: (subject.site_id, subject.subject_id))


def manifest_rows(
    subjects: Iterable[AbideSubject],
    derivative: AbideDerivative = AbideDerivative(),
) -> list[dict[str, object]]:
    """Convert subject records to stable, URL-bearing manifest rows."""

    rows: list[dict[str, object]] = []
    for subject in subjects:
        row = asdict(subject)
        row["diagnosis_name"] = subject.diagnosis_name
        row["derivative_url"] = build_derivative_url(subject.file_id, derivative)
        rows.append(row)
    return rows


def write_manifest(
    subjects: Sequence[AbideSubject],
    output_path: Path,
    derivative: AbideDerivative = AbideDerivative(),
) -> str:
    """Write a stable CSV manifest and return its SHA-256 digest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(subjects, derivative)
    if not rows:
        raise ValueError("Cannot write an empty ABIDE manifest")

    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def summarize(subjects: Iterable[AbideSubject]) -> dict[str, object]:
    """Return deterministic cohort counts for logs and smoke tests."""

    cohort = tuple(subjects)
    return {
        "subjects": len(cohort),
        "sites": len({subject.site_id for subject in cohort}),
        "diagnoses": dict(sorted(Counter(s.diagnosis_name for s in cohort).items())),
    }


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(value: str | None) -> int | None:
    number = _optional_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _validate_unique_file_ids(subjects: Iterable[AbideSubject]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for subject in subjects:
        if subject.file_id in seen:
            duplicates.add(subject.file_id)
        seen.add(subject.file_id)
    if duplicates:
        raise ValueError(f"Duplicate ABIDE file identifiers: {sorted(duplicates)}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for fetching metadata and generating a local manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/abide-cpac-filt-noglobal-cc200.csv"),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    subjects = parse_phenotypic_csv(fetch_phenotypic_csv(args.timeout))
    digest = write_manifest(subjects, args.output)
    summary = summarize(subjects)
    print(f"Wrote {args.output}")
    print(f"SHA256 {digest}")
    print(
        f"Subjects {summary['subjects']} | Sites {summary['sites']} | "
        f"Diagnoses {summary['diagnoses']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

