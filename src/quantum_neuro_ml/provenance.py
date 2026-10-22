"""Capture immutable run inputs and source before an experiment starts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RunRecord:
    """One run's source snapshot, input receipts, and completion report."""

    def __init__(self, output: Path, configuration: dict[str, object], inputs: Mapping[str, Path]):
        self.output = output
        self.input_paths = dict(inputs)
        self.input_payloads = {name: path.read_bytes() for name, path in self.input_paths.items()}
        self.package = Path(__file__).parent
        self.root = self.package.parent.parent
        self.source_paths = {p.relative_to(self.package).as_posix(): p for p in self.package.rglob("*.py")}
        self.source_hashes = {name: digest(path) for name, path in sorted(self.source_paths.items())}
        self.environment_paths = {name: self.root / name for name in ("pyproject.toml", "uv.lock") if (self.root / name).is_file()}
        self.environment_hashes = {name: digest(path) for name, path in self.environment_paths.items()}
        encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
        configuration = json.loads(encoded)
        self.record = {
            "schema": 1, "state": "running", "started_utc": _now(),
            "configuration": configuration,
            "experiment_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "inputs": {name: {"path": str(path), "sha256": hashlib.sha256(self.input_payloads[name]).hexdigest()}
                       for name, path in self.input_paths.items()},
            "source_sha256": self.source_hashes, "environment_sha256": self.environment_hashes,
        }
        output.mkdir(parents=True, exist_ok=True)
        if (output / "report.json").exists():
            raise ValueError("A completion report already exists; choose a new output directory")
        if (output / "run.json").exists():
            previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
            keys = ("configuration", "inputs", "source_sha256", "environment_sha256")
            if previous.get("state") == "complete" or any(previous.get(key) != self.record[key] for key in keys):
                raise ValueError("Run snapshot already exists or differs; choose a new output directory")
            self.record["previous_attempt_utc"] = previous.get("started_utc")
        snapshot = output / "source.zip"
        if snapshot.exists():
            if not (output / "run.json").exists() or digest(snapshot) != previous.get("source_snapshot_sha256"):
                raise ValueError("Source snapshot receipt differs; choose a new output directory")
        if not snapshot.exists():
            with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
                for name, path in sorted(self.source_paths.items()):
                    archive.writestr(f"src/quantum_neuro_ml/{name}", path.read_bytes())
                for name, path in self.environment_paths.items():
                    archive.writestr(name, path.read_bytes())
                for name, path in self.input_paths.items():
                    if path.suffix.lower() in (".json", ".csv"):
                        archive.writestr(f"inputs/{name}/{path.name}", self.input_payloads[name])
                archive.writestr("experiment.json", encoded)
        self.record["source_snapshot_sha256"] = digest(snapshot)
        self._verify()
        write_json(output / "run.json", self.record)

    def _verify(self) -> None:
        snapshot = self.output / "source.zip"
        if not snapshot.is_file() or digest(snapshot) != self.record["source_snapshot_sha256"]:
            raise RuntimeError("Captured source archive changed during execution")
        for name, path in self.source_paths.items():
            if not path.is_file() or digest(path) != self.source_hashes[name]:
                raise RuntimeError(f"Captured source changed during execution: {name}")
        for name, path in self.environment_paths.items():
            if not path.is_file() or digest(path) != self.environment_hashes[name]:
                raise RuntimeError(f"Environment changed during execution: {name}")
        for name, path in self.input_paths.items():
            if not path.is_file() or digest(path) != self.record["inputs"][name]["sha256"]:
                raise RuntimeError(f"Input changed during execution: {name}")

    def finish(self, report: dict[str, object]) -> dict[str, object]:
        self._verify()
        report.update({
            "source_sha256": self.source_hashes,
            "source_snapshot_sha256": self.record["source_snapshot_sha256"],
            "experiment_sha256": self.record["experiment_sha256"],
            "run_started_utc": self.record["started_utc"], "run_finished_utc": _now(),
        })
        if "uv.lock" in self.environment_hashes:
            report["lockfile_sha256"] = self.environment_hashes["uv.lock"]
        write_json(self.output / "report.json", report)
        self.record.update(state="complete", finished_utc=report["run_finished_utc"], report_sha256=digest(self.output / "report.json"))
        write_json(self.output / "run.json", self.record)
        return report


@contextmanager
def recorded_run(output: Path, configuration: dict[str, object], inputs: Mapping[str, Path]) -> Iterator[RunRecord]:
    run = RunRecord(output, configuration, inputs)
    try:
        yield run
        if run.record["state"] != "complete":
            raise RuntimeError("Run ended without a completion report")
    except BaseException as error:
        run.record.update(state="failed", finished_utc=_now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", run.record)
        raise
