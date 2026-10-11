"""Check regional retention against the baseline and known synthetic local balances."""

import numpy as np
from test_class_response import fixture_data

from ecologyhydro.class_response import evaluate
from ecologyhydro.fine_regional_storage import fit_region, predict


def test_regional_baseline_identity():
    data, _ = fixture_data()
    x = np.r_[[8.0] * 7, [1.2] * 5, 0.2, 10, 0.2]
    np.testing.assert_allclose(
        predict(data, x, [8, 1, 0.2], {}, 3), evaluate(x, data, 3), atol=1e-10
    )


def test_regional_fit_recovers_balance_and_ignores_future():
    data, _ = fixture_data()
    extra = {key: value[data["groups"] // 11 == 0].copy() for key, value in data.items()}
    extra["groups"] += 33
    data = {key: np.r_[value, extra[key]] for key, value in data.items()}
    x = np.r_[[8.0] * 7, [1.2] * 5, 0.2, 10, 0.2]
    target = predict(data, x, [8, 1, 0.2], {3: [12, 0.3]}, 4)
    eligible, keep = np.ones_like(target, bool), np.array([True, True, True, False])
    adjustment = np.zeros_like(target)
    recipe = dict(
        bounds=[[1, 0], [30, 0.5]], starts=[[5, 0.1], [15, 0.25], [29, 0.45]], max_nfev=150
    )
    before = fit_region(data, target, adjustment, eligible, keep, x, 3, recipe)
    np.testing.assert_allclose(before["parameters"], [12, 0.3], rtol=1e-5, atol=1e-6)
    target[-1] = np.nan
    adjustment[-1] = np.nan
    data["p"][data["groups"] // 11 == 3] *= 1000
    after = fit_region(data, target, adjustment, eligible, keep, x, 3, recipe)
    assert before == after
