from pathlib import Path

import pytest

from quantum_neuro_ml.data.abide import (
    AbideDerivative,
    build_derivative_url,
    manifest_rows,
    parse_phenotypic_csv,
    summarize,
    write_manifest,
)


CSV_FIXTURE = """SUB_ID,SITE_ID,FILE_ID,DX_GROUP,AGE_AT_SCAN,SEX
50002,PITT,no_filename,1,16.77,1
50003,PITT,Pitt_0050003,1,24.45,1
50004,NYU,NYU_0050004,2,22.10,2
"""


def test_parse_excludes_rows_without_downloadable_files() -> None:
    subjects = parse_phenotypic_csv(CSV_FIXTURE)

    assert [subject.file_id for subject in subjects] == [
        "NYU_0050004",
        "Pitt_0050003",
    ]
    assert summarize(subjects) == {
        "subjects": 2,
        "sites": 2,
        "diagnoses": {"autism": 1, "control": 1},
    }


def test_builds_expected_public_derivative_url() -> None:
    url = build_derivative_url("Pitt_0050003")

    assert url.endswith(
        "/Outputs/cpac/filt_noglobal/rois_cc200/"
        "Pitt_0050003_rois_cc200.1D"
    )


def test_rejects_unsafe_url_components() -> None:
    with pytest.raises(ValueError, match="Unsafe ABIDE file identifier"):
        build_derivative_url("../../secret")
    with pytest.raises(ValueError, match="Unsafe derivative"):
        AbideDerivative(derivative="../rois_cc200")


def test_manifest_is_stable_and_hashable(tmp_path: Path) -> None:
    subjects = parse_phenotypic_csv(CSV_FIXTURE)
    output = tmp_path / "manifest.csv"

    first_digest = write_manifest(subjects, output)
    first_bytes = output.read_bytes()
    second_digest = write_manifest(subjects, output)

    assert first_digest == second_digest
    assert first_bytes == output.read_bytes()
    assert manifest_rows(subjects)[0]["diagnosis_name"] == "control"

