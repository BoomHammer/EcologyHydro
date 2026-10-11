"""Verify local refinement preserves upstream parameters and excludes future data."""

import numpy as np
from test_class_response import fixture_data

from ecologyhydro.class_response import evaluate
from ecologyhydro.independent_local_balance import fit_local_storage, fit_reach_storage, reach_data


def sample():
    data, _ = fixture_data()
    first = data["groups"] // 11 == 0
    extra = {key: values[first].copy() for key, values in data.items()}
    extra["groups"] += 33
    return {key: np.r_[values, extra[key]] for key, values in data.items()}


def test_local_response_equals_cumulative_station_difference():
    data = sample()
    x = np.r_[[9] * 7, [1.2] * 5, 0.2, 10, 0.23]
    whole = evaluate(x, data, 4)
    local = evaluate(x, reach_data(data), 4)
    np.testing.assert_allclose(local[:, 2], whole[:, 2] - whole[:, 1], atol=1e-10)


def test_local_fit_preserves_other_parameters_and_excludes_future():
    data = sample()
    seed = np.r_[[9] * 7, [1.2] * 5, 0.2, 10, 0.23]
    target = evaluate(seed, data, 4)
    adjustment = np.zeros_like(target)
    eligible = np.ones_like(target, dtype=bool)
    keep = np.array([True, True, True, False])
    recipe = dict(bounds=[[1, 0], [30, 100]], starts=[[5, 10], [20, 30], [29, 50]], max_nfev=150)
    before = fit_local_storage(data, target, adjustment, eligible, keep, seed, recipe)
    frozen = np.ones(15, dtype=bool)
    frozen[[2, 13]] = False
    np.testing.assert_array_equal(np.asarray(before["parameters"])[frozen], seed[frozen])
    assert before["local_training_rmse"] < 1e-5
    data["p"][data["groups"] // 11 == 3] *= 100
    target[-1] *= 100
    adjustment[-1] = np.nan
    after = fit_local_storage(data, target, adjustment, eligible, keep, seed, recipe)
    assert before == after


def test_all_reaches_recover_water_balance_without_repeating_ltg_loss():
    data = sample()
    seed = np.r_[np.arange(5, 12), [1.2] * 5, 0.2, 10, 0.23]
    target = evaluate(seed, data, 4)
    adjustment = np.zeros_like(target)
    valid = np.ones_like(target, dtype=bool)
    keep = np.array([True, True, True, False])
    local = dict(parameters=seed.tolist(), seed_parameters=seed.tolist())
    result = fit_reach_storage(data, target, adjustment, valid, keep, local, dict(max_nfev=150))
    np.testing.assert_allclose(result["parameters"], seed, atol=1e-4)
