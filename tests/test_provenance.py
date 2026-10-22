import json
import zipfile
from pathlib import Path

import pytest

from quantum_neuro_ml.provenance import recorded_run


def test_run_captures_sources_and_inputs_before_success(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("original")
    output = tmp_path / "run"
    with recorded_run(output, {"experiment": "test"}, {"input": source}) as run:
        initial = json.loads((output / "run.json").read_text())
        assert initial["state"] == "running"
        assert not (output / "report.json").exists()
        with zipfile.ZipFile(output / "source.zip") as archive:
            assert "src/quantum_neuro_ml/provenance.py" in archive.namelist()
        report = run.finish({"result": 1})
    final = json.loads((output / "run.json").read_text())
    assert final["state"] == "complete"
    assert report["source_sha256"] == initial["source_sha256"]
    assert report["run_started_utc"] == initial["started_utc"]
    with pytest.raises(ValueError, match="completion report already exists"):
        with recorded_run(output, {"experiment": "test"}, {"input": source}):
            pass


def test_changed_input_prevents_completion_and_records_failure(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("original")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Input changed"):
        with recorded_run(output, {"experiment": "test"}, {"input": source}) as run:
            source.write_text("changed")
            run.finish({"result": 1})
    assert not (output / "report.json").exists()
    assert json.loads((output / "run.json").read_text())["state"] == "failed"


def test_changed_source_cannot_be_reported_as_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import quantum_neuro_ml.provenance as provenance
    package = tmp_path / "src" / "quantum_neuro_ml"
    package.mkdir(parents=True)
    source = package / "provenance.py"
    source.write_text("initial source")
    monkeypatch.setattr(provenance, "__file__", str(source))
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Captured source changed"):
        with recorded_run(output, {"experiment": "test"}, {}) as run:
            source.write_text("different source")
            run.finish({"result": 1})
    assert not (output / "report.json").exists()
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert archive.read("src/quantum_neuro_ml/provenance.py") == b"initial source"


def test_changed_source_archive_prevents_completion(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="source archive changed"):
        with recorded_run(output, {"experiment": "test"}, {}) as run:
            (output / "source.zip").write_bytes(b"damaged archive")
            run.finish({"result": 1})
    assert not (output / "report.json").exists()
