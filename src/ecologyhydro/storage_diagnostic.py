"""Strict annual-state accounting for the conditional soil/snow diagnostic."""

import numpy as np

from ecologyhydro.regional_calibration import ENDPOINTS


def cumulative_storage(rows, domain):
    """Sum soil and snow increment changes once, then accumulate to common endpoints."""
    changes = np.zeros((5, 11))
    expected = {
        (year, zone, variable)
        for year in range(2019, 2024)
        for zone in range(1, 12)
        for variable in ("soil", "swe")
    }
    seen = set()
    for row in rows:
        if row["domain"] != domain:
            continue
        key = int(row["year"]), int(row["zone"]), row["variable"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicated storage account")
        seen.add(key)
        amount = float(row["delta_storage_1e8_m3"])
        if not np.isfinite(amount):
            raise ValueError("Invalid storage account")
        changes[key[0] - 2019, key[1] - 1] += amount
    if seen != expected:
        raise ValueError("Missing storage account")
    return changes.cumsum(axis=1)[:, ENDPOINTS]
