"""Check physical regional maps, missing propagation, analytic gradients and temporal isolation."""

import numpy as np
import pytest

from ecologyhydro.long_record import MACRO_MAP, REACH_MAP, number, scores
from ecologyhydro.regional_transfer import evaluate, fit_transfer, pooling
from scripts.test.test_regional_calibration import sample_data


def test_missing_is_unknown_and_propagates_only_downstream():
    assert np.isnan(number(""))
    assert number("0") == 0
    known = np.array([[2.0, 3.0, np.nan, 7.0, 8.0, 9.0, 10.0]])
    cumulative = known.cumsum(axis=1)
    np.testing.assert_array_equal(cumulative[0, :2], [2, 5])
    assert np.isnan(cumulative[0, 2:]).all()
    with pytest.raises(ValueError):
        scores(np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1), dtype=bool))


def test_geographic_macro_regions_are_not_the_old_arbitrary_three_regions():
    np.testing.assert_array_equal(MACRO_MAP, [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(REACH_MAP, [0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 6])


@pytest.mark.parametrize("model,n_z", [("macro3", 3), ("reach7", 7), ("pooled7", 7)])
def test_analytic_fu_derivatives_match_finite_differences(model, n_z):
    data = sample_data()
    # Exercise both nonvegetated and capped-omega cells without sitting on a kink.
    data["veg"][::5] = 0
    data["awc"][::7] = 2000
    x = np.r_[np.linspace(3, 20, n_z), 1.1, 2.5]
    _, jac = evaluate(x, data, model, 4, jacobian=True)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = 1e-5
        numeric = (evaluate(x + delta, data, model, 4) - evaluate(x - delta, data, model, 4)) / 2e-5
        np.testing.assert_allclose(jac[:, :, i], numeric, atol=1e-7, rtol=2e-5)
    reg, reg_jac = pooling(x, model)
    if len(reg):
        for i in range(len(x)):
            delta = np.zeros_like(x)
            delta[i] = 1e-5
            numeric = (pooling(x + delta, model)[0] - pooling(x - delta, model)[0]) / 2e-5
            np.testing.assert_allclose(reg_jac[:, i], numeric, atol=1e-8)


def test_future_observations_accounts_and_climate_cannot_change_training_fit():
    data = sample_data()
    truth = [4, 8, 12, 1.2, 1.0]
    target = evaluate(truth, data, "macro3", 4)
    adjustment = np.zeros((4, 7))
    eligible = np.ones((4, 7), dtype=bool)
    keep = np.array([True, True, True, False])
    # One missing upstream account invalidates dependent stations, not the whole year.
    target[1, 3:] = np.nan
    adjustment[1, 3:] = np.nan
    eligible[1, 3:] = False
    original = fit_transfer(data, target, adjustment, eligible, keep, "macro3")
    target[-1] *= 100
    adjustment[-1] = 99999
    altered = {k: v.copy() for k, v in data.items()}
    altered["p"][altered["groups"] // 11 == 3] *= 3
    poisoned = fit_transfer(altered, target, adjustment, eligible, keep, "macro3")
    np.testing.assert_array_equal(original["parameters"], poisoned["parameters"])
    assert original["observations"] == 17


def test_pooling_preserves_differences_between_upper_middle_and_lower_groups():
    x = np.array([5, 5, 5, 12, 12, 12, 20, 1.2, 15.0])
    np.testing.assert_allclose(pooling(x, "pooled7")[0], 0, atol=1e-16)
