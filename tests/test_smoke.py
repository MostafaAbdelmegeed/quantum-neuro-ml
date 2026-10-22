import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

from quantum_neuro_ml.data.abide import AbideSubject, parse_phenotypic_csv
from quantum_neuro_ml.data.smoke import download_derivative, run_smoke, select_smoke_cohort


METADATA = """SUB_ID,SITE_ID,FILE_ID,DX_GROUP
1,AAA,AAA_0000001,1
2,AAA,AAA_0000002,1
3,AAA,AAA_0000003,2
4,AAA,AAA_0000004,2
5,BBB,BBB_0000005,1
6,BBB,BBB_0000006,1
7,BBB,BBB_0000007,2
8,BBB,BBB_0000008,2
"""


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


def test_selection_is_balanced_and_independent_of_input_order() -> None:
    subjects = parse_phenotypic_csv(METADATA)
    cohort = select_smoke_cohort(subjects, ("BBB", "AAA"))
    assert cohort == select_smoke_cohort(list(reversed(subjects)), ("AAA", "BBB"))
    assert len(cohort) == 8
    assert select_smoke_cohort(subjects, ("AAA", "BBB"), 1) == tuple(subjects[::2])


@pytest.mark.parametrize("sites, per_group", [
    (("AAA",), 2),
    (("AAA", "AAA"), 2),
    (("AAA", "BBB"), 0),
    (("AAA", "BBB"), 3),
    (("AAA", "MISSING"), 1),
])
def test_selection_rejects_invalid_or_unavailable_groups(sites: tuple[str, ...], per_group: int) -> None:
    with pytest.raises(ValueError):
        select_smoke_cohort(parse_phenotypic_csv(METADATA), sites, per_group)


def test_selection_rejects_duplicate_participants() -> None:
    subjects = parse_phenotypic_csv(METADATA)
    with pytest.raises(ValueError, match="Duplicate participant"):
        select_smoke_cohort(subjects + subjects[:1], ("AAA", "BBB"))


def test_smoke_downloads_then_rebuilds_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = tmp_path / "phenotypic.csv"
    metadata.write_text(METADATA, encoding="utf-8")
    data = np.random.default_rng(42).normal(size=(10, 200))
    buffer = io.StringIO()
    np.savetxt(buffer, data, header="\t".join(f"#{i}" for i in range(1, 201)), comments="")
    payload = buffer.getvalue().encode()
    requests: list[str] = []

    def fetch(request: object, **kwargs: object) -> Response:
        requests.append(request.full_url)
        return Response(payload)

    monkeypatch.setattr("quantum_neuro_ml.data.smoke.urllib.request.urlopen", fetch)
    options = dict(metadata_path=metadata, raw_dir=tmp_path / "raw", output_dir=tmp_path / "out",
                   sites=("AAA", "BBB"), held_out_site="BBB")
    report = run_smoke(**options)
    assert len(requests) == 8
    assert report["subjects"] == 8
    assert report["train_subjects"] == report["test_subjects"] == 4
    assert report["diagnoses"] == {"autism": 4, "control": 4}
    assert report["metadata_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert all(r["source_sha256"] == hashlib.sha256(payload).hexdigest() for r in report["records"])
    with np.load(tmp_path / "out" / "connectivity.npz", allow_pickle=False) as archive:
        assert archive["correlation"].shape == (8, 200, 200)
        assert archive["features"].shape == (8, 19900)
        np.testing.assert_array_equal(archive["labels"], [1, 1, 0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(archive["is_test"], [False] * 4 + [True] * 4)
        assert set(archive["site_ids"][archive["is_test"]]) == {"BBB"}
        assert set(archive["site_ids"][~archive["is_test"]]) == {"AAA"}
        assert np.isfinite(archive["features"]).all()
        expected = np.corrcoef(data, rowvar=False)
        np.testing.assert_allclose(archive["correlation"][0], expected, atol=1e-14)
        edge_i, edge_j = np.triu_indices(200, k=1)
        np.testing.assert_allclose(archive["features"][0], np.arctanh(expected[edge_i, edge_j]))

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Offline rebuild attempted network access")

    monkeypatch.setattr("quantum_neuro_ml.data.smoke.urllib.request.urlopen", no_network)
    second_report = run_smoke(**options, offline=True)
    assert report == second_report


def test_download_rejects_cache_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subject = parse_phenotypic_csv(METADATA)[0]
    monkeypatch.setattr("quantum_neuro_ml.data.smoke.urllib.request.urlopen",
                        lambda *a, **kw: Response(b"original bytes"))
    path, digest = download_derivative(subject, tmp_path)
    assert digest == hashlib.sha256(b"original bytes").hexdigest()
    path.write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="checksum or URL mismatch"):
        download_derivative(subject, tmp_path, offline=True)


def test_download_rejects_incomplete_cache_and_missing_offline_data(tmp_path: Path) -> None:
    subject = parse_phenotypic_csv(METADATA)[0]
    with pytest.raises(ValueError, match="cached derivative"):
        download_derivative(subject, tmp_path, offline=True)
    path = tmp_path / f"{subject.file_id}_rois_cc200.1D"
    path.write_bytes(b"unverified")
    with pytest.raises(ValueError, match="Incomplete download cache"):
        download_derivative(subject, tmp_path)


def test_download_does_not_cache_a_truncated_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    response = Response(b"truncated")
    response.headers["Content-Length"] = "100"
    monkeypatch.setattr("quantum_neuro_ml.data.smoke.urllib.request.urlopen", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="Incomplete ROI derivative response"):
        download_derivative(parse_phenotypic_csv(METADATA)[0], tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_rejects_unsafe_identifier_before_writing(tmp_path: Path) -> None:
    subject = AbideSubject(1, "AAA", "../../outside", 1, None, None)
    with pytest.raises(ValueError, match="Unsafe ABIDE file identifier"):
        download_derivative(subject, tmp_path)
    assert not list(tmp_path.iterdir())


def test_offline_run_requires_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cached metadata"):
        run_smoke(metadata_path=tmp_path / "missing.csv", raw_dir=tmp_path / "raw",
                  output_dir=tmp_path / "out", offline=True)
