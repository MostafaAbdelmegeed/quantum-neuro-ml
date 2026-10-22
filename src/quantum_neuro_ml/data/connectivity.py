"""Validate PCP ROI time series and construct subject-level connectivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


FISHER_CLIP = 1e-7


def load_roi_timeseries(path: Path, expected_rois: int = 200) -> NDArray[np.float64]:
    """Read a PCP .1D file, checking that ROI columns have the expected order."""

    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline().split()
    expected_header = [f"#{index}" for index in range(1, expected_rois + 1)]
    if header != expected_header:
        raise ValueError(f"{path.name}: expected ordered ROI labels 1..{expected_rois}")

    timeseries = np.loadtxt(path, comments="#", ndmin=2, encoding="utf-8-sig")
    if timeseries.shape[1] != expected_rois:
        raise ValueError(f"{path.name}: expected {expected_rois} ROI columns")
    return timeseries


def functional_connectivity(
    timeseries: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Pearson correlations and Fisher-z edges for one participant.

    Rows are timepoints; columns are ROIs. This operation uses only this
    participant's scan. Both outputs are symmetric; correlation has a unit
    diagonal and Fisher-z has a zero diagonal. Off-diagonal correlations are
    clipped to +/- (1 - FISHER_CLIP) before arctanh to keep perfect correlations
    finite. Constant or non-finite signals cause an error rather than imputation.
    """

    values = np.asarray(timeseries, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError("Expected at least 3 timepoints and 2 ROI columns")
    if not np.isfinite(values).all():
        raise ValueError("ROI time series contain non-finite values")
    if np.any(np.ptp(values, axis=0) == 0):
        raise ValueError("ROI time series contain a constant ROI")

    correlation = np.corrcoef(values, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("ROI correlation calculation produced non-finite values")
    correlation = np.clip((correlation + correlation.T) / 2, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    fisher_z = np.arctanh(np.clip(correlation, -1 + FISHER_CLIP, 1 - FISHER_CLIP))
    np.fill_diagonal(fisher_z, 0.0)
    return correlation, fisher_z
