from dataclasses import replace

import pytest

from quantum_neuro_ml.data.abide import AbideSubject
from quantum_neuro_ml.data.splits import leave_one_site_out


def subject(file_id: str, site_id: str, diagnosis: int) -> AbideSubject:
    return AbideSubject(
        subject_id=int(file_id.rsplit("_", 1)[-1]),
        site_id=site_id,
        file_id=file_id,
        diagnosis=diagnosis,
        age_at_scan=None,
        sex=None,
    )


COHORT = (
    subject("PITT_1", "PITT", 1),
    subject("PITT_2", "PITT", 2),
    subject("NYU_3", "NYU", 1),
    subject("NYU_4", "NYU", 2),
)


def test_leave_one_site_out_is_disjoint() -> None:
    split = leave_one_site_out(COHORT, "NYU")

    assert {record.site_id for record in split.train} == {"PITT"}
    assert {record.site_id for record in split.test} == {"NYU"}
    assert {record.file_id for record in split.train}.isdisjoint(
        record.file_id for record in split.test
    )


def test_leave_one_site_out_rejects_unknown_site() -> None:
    with pytest.raises(ValueError, match="absent"):
        leave_one_site_out(COHORT, "UNKNOWN")


def test_leave_one_site_out_requires_multiple_sites() -> None:
    with pytest.raises(ValueError, match="at least two sites"):
        leave_one_site_out(COHORT[:2], "PITT")


def test_rejects_participant_leakage_under_different_file_ids() -> None:
    duplicate = replace(COHORT[0], site_id="NYU", file_id="NYU_other_acquisition")
    with pytest.raises(ValueError, match="Participant leakage"):
        leave_one_site_out((*COHORT, duplicate), "NYU")


def test_repeat_samples_from_one_acquisition_centre_are_held_out_together() -> None:
    cohort = (subject("UCLA_1", "UCLA_1", 1), subject("UCLA_2", "UCLA_2", 2),
              subject("NYU_3", "NYU", 1))
    split = leave_one_site_out(cohort, "UCLA_1")
    assert split.held_out_site == "UCLA"
    assert {s.site_id for s in split.test} == {"UCLA_1", "UCLA_2"}
    assert {s.site_id for s in split.train} == {"NYU"}

