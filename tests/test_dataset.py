from pathlib import Path

import numpy as np
import pytest

from quantum_neuro_ml.data.dataset import build_dataset
from quantum_neuro_ml.data.cohort import CohortPolicy, main as audit_main
from quantum_neuro_ml.baselines import load_research_dataset, main as baseline_main
from test_cohort import metadata_row, write_fixture


def test_build_records_signal_exclusions_and_rechecks_site_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [metadata_row(4 * group + index + 1, chr(65 + group), 1 + index // 2)
            for group in range(5) for index in range(4)]
    metadata, policy_path, _ = write_fixture(tmp_path, rows, per_diagnosis=2)
    def download(subject, raw_dir, **kwargs):
        path = tmp_path / f"{subject.file_id}.1D"
        rng = np.random.default_rng(subject.subject_id)
        signals = rng.normal(size=(6, 200))
        if subject.subject_id == 1:
            signals[:, 0] = 1
        np.savetxt(path, signals, header="\t".join(f"#{i}" for i in range(1, 201)), comments="")
        return path, "fixture-checksum"
    monkeypatch.setattr("quantum_neuro_ml.data.dataset.download_derivative", download)
    output = tmp_path / "research"
    report = build_dataset(metadata_path=metadata, policy_path=policy_path,
                           output_dir=output, offline=True, workers=2)
    assert report["included"]["subjects"] == 16
    assert report["included"]["acquisition_groups"] == 4
    import json
    audit = json.loads((output / "audit.json").read_text())
    assert audit["first_exclusion_reason"] == {"invalid_roi_signal": 1, "site_below_minimum_after_scan_validation": 3}
    with np.load(output / "connectivity.npz", allow_pickle=False) as archive:
        assert archive["features"].shape == (16, 19900)
        assert set(archive["site_groups"]) == {"B", "C", "D", "E"}
        assert np.isfinite(archive["features"]).all()
    with pytest.raises(ValueError, match="already exist"):
        build_dataset(metadata_path=metadata, policy_path=policy_path, output_dir=output)
    with pytest.raises(ValueError, match="contain a research dataset"):
        audit_main(["--metadata", str(metadata), "--policy", str(policy_path), "--output-dir", str(output)])
    arrays, _ = load_research_dataset(output / "connectivity.npz", CohortPolicy.load(policy_path))
    assert len(arrays["labels"]) == 16
    model_output = tmp_path / "models"
    assert baseline_main(["--dataset", str(output / "connectivity.npz"), "--policy", str(policy_path),
                          "--output-dir", str(model_output)]) == 0
    model_report = json.loads((model_output / "report.json").read_text())
    assert model_report["subjects"] == 16
    assert len((model_output / "predictions.csv").read_text().splitlines()) == 49
    assert set(model_report["summaries"]) == {"dummy", "logistic", "rbf_svm"}
    with (output / "connectivity.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_research_dataset(output / "connectivity.npz", CohortPolicy.load(policy_path))


@pytest.mark.parametrize("field, value, message", [
    ("metadata_sha256", "0" * 64, "metadata fingerprint"),
    ("manifest_sha256", "0" * 64, "manifest checksum"),
    ("derivative", "different/strategy/atlas", "derivative differs"),
])
def test_loader_rejects_inconsistent_report_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   field: str, value: str, message: str) -> None:
    import json
    import hashlib
    from quantum_neuro_ml.data.cohort import CohortPolicy
    # Report validation must fail before the archive is read.
    policy = CohortPolicy.load(Path("config/cohort-v1.json"))
    report = {"purpose": "research_dataset", "policy_sha256": policy.sha256,
              "metadata_sha256": policy.metadata_sha256, "manifest_sha256": "unused",
              "derivative": "cpac/filt_noglobal/rois_cc200"}
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("unused")
    report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report[field] = value
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        load_research_dataset(tmp_path / "connectivity.npz", policy)


@pytest.mark.parametrize("array_name, bad_value, message", [
    ("age", float("nan"), "Non-finite research covariates"),
    ("sex", 3, "age or sex inclusion rules"),
    ("mean_fd", 0.2, "motion violates"),
    ("labels", 0, "manifest rows do not align"),
])
def test_loader_validates_covariates_and_manifest_alignment(tmp_path: Path, array_name: str,
                                                          bad_value: float, message: str) -> None:
    import json
    import hashlib
    from quantum_neuro_ml.data.cohort import audit_metadata, write_audit
    rows = [metadata_row(2 * group + index + 1, chr(65 + group), 1 + index)
            for group in range(4) for index in range(2)]
    metadata, _, policy = write_fixture(tmp_path, rows)
    subjects, decisions, audit = audit_metadata(metadata, policy)
    write_audit(tmp_path, subjects, decisions, audit)
    labels = np.array([1, 0] * 4)
    arrays = dict(features=np.zeros((8, 19900)), labels=labels,
                  subject_ids=np.arange(1, 9), site_ids=np.repeat(list("ABCD"), 2),
                  site_groups=np.repeat(list("ABCD"), 2), age=np.full(8, 20.0),
                  sex=np.ones(8), mean_fd=np.full(8, 0.1))
    if array_name == "labels":
        arrays["labels"] = 1 - labels
    else:
        arrays[array_name][0] = bad_value
    path = tmp_path / "connectivity.npz"
    np.savez_compressed(path, **arrays)
    report = dict(purpose="research_dataset", policy_sha256=policy.sha256,
                  metadata_sha256=policy.metadata_sha256, manifest_sha256=audit["manifest_sha256"],
                  derivative="cpac/filt_noglobal/rois_cc200", feature_count=19900,
                  dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        load_research_dataset(path, policy)


def test_transport_failure_stops_build_instead_of_excluding_participant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [metadata_row(2 * group + index + 1, chr(65 + group), 1 + index)
            for group in range(4) for index in range(2)]
    metadata, policy_path, _ = write_fixture(tmp_path, rows)
    def broken_download(*args, **kwargs):
        raise OSError("temporary server failure")
    monkeypatch.setattr("quantum_neuro_ml.data.dataset.download_derivative", broken_download)
    output = tmp_path / "research"
    with pytest.raises(OSError, match="server failure"):
        build_dataset(metadata_path=metadata, policy_path=policy_path, output_dir=output, workers=1)
    assert not (output / "report.json").exists()
    assert not (output / "connectivity.npz").exists()
