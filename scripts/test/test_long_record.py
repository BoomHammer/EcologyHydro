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
@pytest.mark.parametrize("kc_cap", [1.3, None])
def test_analytic_fu_derivatives_match_finite_differences(model, n_z, kc_cap):
    data = sample_data()
    # Exercise both nonvegetated and capped-omega cells without sitting on a kink.
    data["veg"][::5] = 0
    data["awc"][::7] = 2000
    x = np.r_[np.linspace(3, 20, n_z), 1.8, 2.5]
    _, jac = evaluate(x, data, model, 4, jacobian=True, kc_cap=kc_cap)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = 1e-5
        numeric = (
            evaluate(x + delta, data, model, 4, kc_cap=kc_cap)
            - evaluate(x - delta, data, model, 4, kc_cap=kc_cap)
        ) / 2e-5
        np.testing.assert_allclose(jac[:, :, i], numeric, atol=1e-7, rtol=2e-5)
    reg, reg_jac = pooling(x, model)
    if len(reg):
        for i in range(len(x)):
            delta = np.zeros_like(x)
            delta[i] = 1e-5
            numeric = (pooling(x + delta, model)[0] - pooling(x - delta, model)[0]) / 2e-5
            np.testing.assert_allclose(reg_jac[:, i], numeric, atol=1e-8)


@pytest.mark.parametrize("kc_cap", [1.3, None])
def test_future_observations_accounts_and_climate_cannot_change_training_fit(kc_cap):
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
    original = fit_transfer(data, target, adjustment, eligible, keep, "macro3", kc_cap=kc_cap)
    target[-1] *= 100
    adjustment[-1] = 99999
    altered = {k: v.copy() for k, v in data.items()}
    altered["p"][altered["groups"] // 11 == 3] *= 3
    poisoned = fit_transfer(altered, target, adjustment, eligible, keep, "macro3", kc_cap=kc_cap)
    np.testing.assert_array_equal(original["parameters"], poisoned["parameters"])
    assert original["observations"] == 17


def test_pooling_preserves_differences_between_upper_middle_and_lower_groups():
    x = np.array([5, 5, 5, 12, 12, 12, 20, 1.2, 15.0])
    np.testing.assert_allclose(pooling(x, "pooled7")[0], 0, atol=1e-16)


@pytest.mark.parametrize("kc_cap", [1.3, None])
def test_kc_policy_matches_official_kernel(kc_cap):
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    from ecologyhydro.long_record import ENDPOINTS

    data = sample_data()
    data["veg"][::5] = 0
    x = np.r_[np.linspace(3, 20, 7), 1.9, 2.5]
    kc = np.where(
        data["veg"] == 1,
        np.minimum(data["kc"] * x[-2], np.inf if kc_cap is None else kc_cap),
        data["kc"],
    )
    expected_depth = np.zeros(len(kc))
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    for ri in range(7):
        mask = REACH_MAP[data["zone"]] == ri
        fraction = fractp_op(
            kc[mask],
            data["et"][mask],
            data["p"][mask],
            np.full(mask.sum(), 1000),
            np.full(mask.sum(), 1000),
            data["awc"][mask] / 1000,
            data["veg"][mask],
            nodata,
            x[ri],
        )
        expected_depth[mask] = (1 - fraction) * data["p"][mask]
    expected = np.bincount(data["groups"], weights=expected_depth * data["weights"])
    expected = expected.reshape(4, 11).cumsum(axis=1)[:, ENDPOINTS]
    expected[:, 2:] -= x[-1]
    np.testing.assert_allclose(evaluate(x, data, "reach7", 4, kc_cap=kc_cap), expected, atol=1e-5)
