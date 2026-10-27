"""Check that the published aggregate still points to its audited evidence."""

import hashlib
import json
from pathlib import Path


def test_public_aggregate_receipts():
    root = Path(__file__).resolve().parents[1]
    aggregate = json.loads((root / "docs/results/aggregate.json").read_text())
    audit_path = root / "docs/results/artifact-audit.json"
    audit = json.loads(audit_path.read_text())
    assert aggregate["audit_report_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert aggregate["plot_script_sha256"] == hashlib.sha256(
        (root / "scripts/plot_results.py").read_bytes()
    ).hexdigest()
    assert audit["audit"] == "passed"
    for field, path in [
        ("script_sha256", "scripts/audit_results.py"),
        ("lockfile_sha256", "uv.lock"),
    ]:
        assert audit[field] == hashlib.sha256((root / path).read_bytes()).hexdigest()
    audited_reports = {row["report_sha256"] for row in audit["experiments"]}
    audited_reports.add(audit["permutation_test"]["report_sha256"])
    assert {row["report_sha256"] for row in aggregate["experiments"].values()} <= audited_reports
