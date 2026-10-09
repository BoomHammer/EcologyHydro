"""Check nested station accounting and training-only bounded parameter estimation."""

import numpy as np
import pytest
from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

from scripts.calibrate_major_bias import fit, predict


def synthetic_data():
    rng = np.random.default_rng(42)
    return {
        "p": rng.uniform(300, 1100, 44),
        "et": rng.uniform(400, 1000, 44),
        "kc": rng.uniform(0.4, 0.8, 44),
        "awc": rng.uniform(20, 120, 44),
        "veg": np.ones(44),
        "weights": np.full(44, 0.01),
        "groups": np.arange(44),
    }


def test_nested_station_prediction_matches_official():
    data = synthetic_data()
    data["veg"][::3] = 0
    z, multiplier = 12, 1.8
    kc = np.where(data["veg"] == 1, np.minimum(data["kc"] * multiplier, 1.3), data["kc"])
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    fraction = fractp_op(
        kc,
        data["et"],
        data["p"],
        np.full(44, 1000),
        np.full(44, 1000),
        data["awc"] / 1000,
        data["veg"],
        nodata,
        z,
    )
    zone_yield = data["p"] * (1 - fraction) * data["weights"]
    np.testing.assert_allclose(
        predict([z, multiplier], data), np.cumsum(zone_yield.reshape(4, 11), axis=1), atol=1e-5
    )


def test_fit_recovers_known_parameters_and_ignores_withheld_targets():
    data = synthetic_data()
    target = predict([12, 1.4], data)
    keep = np.ones((4, 11), dtype=bool)
    keep[3] = False
    altered = target.copy()
    altered[3] *= 100
    parameters, score = fit(data, altered, keep)
    assert parameters == pytest.approx([12, 1.4], abs=0.001)
    assert score < 1e-6
