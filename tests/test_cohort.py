import csv
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from quantum_neuro_ml.data.cohort import CohortPolicy, QC_COLUMNS, audit_metadata


def metadata_row(identifier: int, site: str = "A", diagnosis: int = 1, **overrides: str) -> dict[str, str]:
    row = dict(SUB_ID=str(identifier), FILE_ID=f"{site}_{identifier}", SITE_ID=site,
               DX_GROUP=str(diagnosis), AGE_AT_SCAN="20", SEX="1", func_mean_fd="0.1")
    row.update({key: "OK" for key in QC_COLUMNS})
    row.update(overrides)
    return row


def write_fixture(tmp_path: Path, rows: list[dict[str, str]], *, per_diagnosis: int = 1) -> tuple[Path, Path, CohortPolicy]:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode()
    metadata = tmp_path / "metadata.csv"
    metadata.write_bytes(payload)
    policy = CohortPolicy("fixture-v1", hashlib.sha256(payload).hexdigest(), 0.2, 5,
                          2 * per_diagnosis, per_diagnosis, QC_COLUMNS)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(asdict(policy)), encoding="utf-8")
    return metadata, policy_path, policy


def test_boundaries_missing_values_and_manual_qc_are_explicit(tmp_path: Path) -> None:
    rows = [metadata_row(1), metadata_row(2, diagnosis=2)]
    rows += [metadata_row(3, func_mean_fd="0.2"), metadata_row(4, func_mean_fd="nan"),
             metadata_row(5, AGE_AT_SCAN="-9999"), metadata_row(6, qc_func_rater_2="maybe"),
             metadata_row(7, FILE_ID="no_filename"), metadata_row(8, SEX="inf")]
    path, _, policy = write_fixture(tmp_path, rows)
    subjects, decisions, report = audit_metadata(path, policy)
    assert [s.subject.subject_id for s in subjects] == [1, 2]
    reasons = {d["subject_id"]: d["reasons"] for d in decisions}
    assert reasons[3] == ["motion_at_or_above_threshold"]
    assert reasons[4] == ["missing_or_invalid_motion"]
    assert reasons[5] == ["missing_or_invalid_covariates"]
    assert reasons[6] == ["manual_qc_not_all_ok"]
    assert reasons[7] == ["no_derivative"]
    assert reasons[8] == ["missing_or_invalid_covariates"]
    assert sum(report["first_exclusion_reason"].values()) + len(subjects) == len(rows)


def test_site_minimum_combines_repeat_samples_and_requires_both_diagnoses(tmp_path: Path) -> None:
    rows = [metadata_row(1, "UCLA_1"), metadata_row(2, "UCLA_2", 2),
            metadata_row(3, "OTHER"), metadata_row(4, "OTHER")]
    path, _, policy = write_fixture(tmp_path, rows)
    subjects, decisions, report = audit_metadata(path, policy)
    assert len(subjects) == 2
    assert report["included"]["acquisition_groups"] == 1
    assert report["included"]["site_labels"] == 2
    assert decisions[-1]["reasons"] == ["site_below_minimum"]


def test_audit_is_independent_of_metadata_row_order(tmp_path: Path) -> None:
    rows = [metadata_row(1), metadata_row(2, diagnosis=2), metadata_row(3, func_mean_fd="0.3")]
    path, _, policy = write_fixture(tmp_path, rows)
    first, first_decisions, _ = audit_metadata(path, policy)
    path, _, policy = write_fixture(tmp_path, list(reversed(rows)))
    second, second_decisions, _ = audit_metadata(path, policy)
    assert first == second
    assert first_decisions == second_decisions


def test_duplicate_participants_and_changed_snapshot_stop_audit(tmp_path: Path) -> None:
    path, _, policy = write_fixture(tmp_path, [metadata_row(1), metadata_row(1, "B", 2)])
    with pytest.raises(ValueError, match="Duplicate participant"):
        audit_metadata(path, policy)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fingerprint"):
        audit_metadata(path, policy)
