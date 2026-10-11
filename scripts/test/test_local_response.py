"""Regional controls retain map identity, other parameters and held-year isolation."""

import numpy as np

from ecologyhydro.local_response import common_response, fit_local, local_data
from ecologyhydro.long_record import REACH_MAP
from ecologyhydro.regional_transfer import evaluate
from scripts.test.test_regional_calibration import sample_data


def test_common_response_retains_inputs_and_uses_matching_soil_capacity():
    data = local_data(sample_data())
    data["code"] = np.arange(len(data["p"])) % 2 + 9
    data["root_depth"] = np.full(len(data["p"]), 1.0)
    data["soil"] = np.full(len(data["p"]), 500.0)
    data["pawc"] = np.full(len(data["p"]), 0.2)
    data["awc"] = np.full(len(data["p"]), 0.2)
    data["veg"][:] = 0
    out = common_response(data, [9], {"kc": 0.3, "lulc_veg": 1, "root_depth": 1000})
    selected = data["code"] == 9
    np.testing.assert_allclose(out["awc"][selected], 100)
    np.testing.assert_allclose(out["awc"][~selected], 0.2)
    assert (REACH_MAP[out["zone"]] == 2).all()
    for key in ("code", "p", "et", "weights", "zone", "groups"):
        np.testing.assert_array_equal(out[key], data[key])
    assert (data["veg"] == 0).all()


def test_local_fit_negative_increment_and_held_year_isolation():
    data = local_data(sample_data())
    seed = [4, 5, 6, 7, 8, 9, 10, 1.2, 10]
    truth = evaluate(seed, data, "reach7", 4)[:, 2]
    known = np.full(4, 15.0)
    observed = truth - known
    keep = np.array([True, True, True, False])
    settings = {"Z_bounds": [1, 30], "L_bounds": [0, 100], "starts": [[5, 10], [20, 30], [29, 50]]}
    original = fit_local(data, observed, known, np.ones(4, bool), keep, seed, settings)
    observed[-1], known[-1] = np.nan, np.nan
    altered = {key: value.copy() for key, value in data.items()}
    altered["p"][altered["groups"] // 11 == 3] *= 100
    poisoned = fit_local(altered, observed, known, np.ones(4, bool), keep, seed, settings)
    np.testing.assert_array_equal(original["parameters"], poisoned["parameters"])
    frozen = [0, 1, 3, 4, 5, 6, 7]
    np.testing.assert_array_equal(np.array(original["parameters"])[frozen], np.array(seed)[frozen])
    assert original["local_rmse"] < 1e-6
