from pathlib import Path

import numpy as np
import pytest

from quantum_neuro_ml.data.connectivity import (
    FISHER_CLIP,
    functional_connectivity,
    load_roi_timeseries,
)


def test_connectivity_preserves_known_positive_and_negative_edges() -> None:
    signal = np.arange(5, dtype=float)
    correlation, fisher_z = functional_connectivity(np.column_stack((signal, signal, -signal)))

    np.testing.assert_allclose(correlation, [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    limit = np.arctanh(1 - FISHER_CLIP)
    np.testing.assert_allclose(fisher_z, [[0, limit, -limit], [limit, 0, -limit], [-limit, -limit, 0]])
    assert np.isfinite(fisher_z).all()


@pytest.mark.parametrize("values, message", [
    (np.ones((4, 3)), "constant"),
    ([[1, 2], [2, float("nan")], [3, 4]], "non-finite"),
    ([[1, 2], [2, float("inf")], [3, 4]], "non-finite"),
    ([[1, 2], [2, 3]], "at least 3 timepoints"),
    (np.arange(5), "at least 3 timepoints"),
])
def test_rejects_invalid_signals(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        functional_connectivity(values)


def test_load_checks_roi_order_and_column_count(tmp_path: Path) -> None:
    path = tmp_path / "roi.1D"
    path.write_text("#1\t#2\n1 2\n2 3\n3 5\n", encoding="utf-8")
    assert load_roi_timeseries(path, expected_rois=2).shape == (3, 2)

    path.write_text("#2\t#1\n1 2\n2 3\n3 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordered ROI labels"):
        load_roi_timeseries(path, expected_rois=2)

    path.write_text("#1\t#2\n1 2 3\n2 3 4\n3 5 6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2 ROI columns"):
        load_roi_timeseries(path, expected_rois=2)
