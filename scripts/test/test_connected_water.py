"""Protect the direction and temporal meaning of the added physical diagnostics."""

import numpy as np
import pytest

from ecologyhydro.storage_diagnostic import cumulative_storage
from scripts.audit_guide_connectivity import trace_exit


def storage_rows():
    return [
        {
            "domain": "candidate",
            "year": year,
            "zone": zone,
            "variable": variable,
            "delta_storage_1e8_m3": 0.0,
        }
        for year in range(2019, 2024)
        for zone in range(1, 12)
        for variable in ("soil", "swe")
    ]


def test_storage_accumulates_space_without_accumulating_years():
    rows = storage_rows()
    for row in rows:
        if row["year"] == 2019 and row["zone"] == 3:
            row["delta_storage_1e8_m3"] = 5 if row["variable"] == "soil" else -2
        if row["year"] == 2020 and row["zone"] == 1 and row["variable"] == "soil":
            row["delta_storage_1e8_m3"] = -7
    actual = cumulative_storage(rows, "candidate")
    np.testing.assert_array_equal(actual[0], [0, 3, 3, 3, 3, 3])
    np.testing.assert_array_equal(actual[1], [-7] * 6)
    np.testing.assert_array_equal(actual[2:], 0)
    np.testing.assert_array_equal(100 - actual[0], [100, 97, 97, 97, 97, 97])
    np.testing.assert_array_equal(100 - actual[1], [107] * 6)


def test_storage_rejects_missing_duplicates_and_nonfinite():
    rows = storage_rows()
    with pytest.raises(ValueError, match="Missing"):
        cumulative_storage(rows[:-1], "candidate")
    with pytest.raises(ValueError, match="duplicated"):
        cumulative_storage(rows + [rows[0]], "candidate")
    rows[0]["delta_storage_1e8_m3"] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        cumulative_storage(rows, "candidate")


def test_d8_trace_distinguishes_polygon_exit_terminal_and_cycle():
    mask = np.array([[1, 1, 0]], dtype=bool)
    direction = np.array([[0, 0, 255]], dtype=np.uint8)
    cells, status = trace_exit(direction, mask, (0, 0))
    assert status == "polygon_exit"
    np.testing.assert_array_equal(cells, [[0, 0], [1, 0], [2, 0]])
    direction[0, 1] = 255
    assert trace_exit(direction, mask, (0, 0))[1] == "terminal"
    direction[0, 1] = 4
    with pytest.raises(ValueError, match="cycle"):
        trace_exit(direction, mask, (0, 0))
