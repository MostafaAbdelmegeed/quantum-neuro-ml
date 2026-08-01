"""Leakage-resistant dataset splitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from quantum_neuro_ml.data.abide import AbideSubject


@dataclass(frozen=True, slots=True)
class SiteHoldoutSplit:
    """A train/test partition with exactly one acquisition site held out."""

    held_out_site: str
    train: tuple[AbideSubject, ...]
    test: tuple[AbideSubject, ...]


def leave_one_site_out(
    subjects: Iterable[AbideSubject],
    held_out_site: str,
) -> SiteHoldoutSplit:
    """Create a deterministic site holdout and validate subject separation."""

    cohort = tuple(subjects)
    if not cohort:
        raise ValueError("Cannot split an empty cohort")

    train = tuple(subject for subject in cohort if subject.site_id != held_out_site)
    test = tuple(subject for subject in cohort if subject.site_id == held_out_site)
    if not test:
        raise ValueError(f"Held-out site is absent from the cohort: {held_out_site}")
    if not train:
        raise ValueError("A site holdout requires at least two sites")

    train_ids = {subject.file_id for subject in train}
    test_ids = {subject.file_id for subject in test}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Subject leakage across site split: {sorted(overlap)}")
    if any(subject.site_id == held_out_site for subject in train):
        raise AssertionError("Held-out site leaked into training records")
    if any(subject.site_id != held_out_site for subject in test):
        raise AssertionError("Test records include a non-held-out site")

    return SiteHoldoutSplit(held_out_site=held_out_site, train=train, test=test)

