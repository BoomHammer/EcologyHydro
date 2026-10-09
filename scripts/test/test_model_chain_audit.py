"""Analytic checks for the independent diagnostic equation."""

import numpy as np
import pytest

from scripts.audit_model_chain import independent_yield


def test_fu_omega_two_has_analytic_solution():
    value = independent_yield(
        *[np.array([v], dtype=float) for v in (100, 100, 1, 250, 1000, 0.3, 1)], z=1
    )
    assert value[0] == pytest.approx(100 * (np.sqrt(2) - 1))


def test_nonvegetated_branch_and_zero_demand():
    p = np.array([100, 100, 100], dtype=float)
    result = independent_yield(
        p, np.array([0, 40, 200]), np.ones(3), p, p, np.full(3, 0.1), np.zeros(3)
    )
    np.testing.assert_allclose(result, [100, 60, 0])


def test_root_limitation_and_evaporative_bound():
    p = np.array([800.0, 800.0])
    output = independent_yield(
        p,
        np.full(2, 500),
        np.full(2, 0.65),
        np.array([300, 1500]),
        np.full(2, 300),
        np.full(2, 0.1),
        np.ones(2),
    )
    assert output[0] == pytest.approx(output[1])
    assert np.all(output >= p - 325)


@pytest.mark.parametrize("z", [5, 30])
@pytest.mark.parametrize("awc_multiplier", [1, 2])
@pytest.mark.parametrize("vegetated_kc_one", [False, True])
def test_factorial_diagnostic_matches_official_invest(z, awc_multiplier, vegetated_kc_one):
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    p = np.array([837, 300, 900, 80, 650], dtype=float)
    et0 = np.array([447, 600, 1100, 300, 0], dtype=float)
    veg = np.array([1, 1, 0, 0, 1], dtype=float)
    kc = np.array([0.65, 0.55, 1.05, 0.05, 0.8])
    if vegetated_kc_one:
        kc = np.where(veg == 1, 1, kc)
    roots = np.array([300, 500, 1, 1, 2000], dtype=float)
    soil = np.array([750, 300, 50, 50, 1500], dtype=float)
    pawc = np.array([0.116, 0.13, 0, 0.1, 0.2]) * awc_multiplier
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    official_aet = p * fractp_op(kc, et0, p, roots, soil, pawc, veg, nodata, z)
    independent_aet = p - independent_yield(p, et0, kc, roots, soil, pawc, veg, z=z)
    np.testing.assert_allclose(independent_aet, official_aet, atol=0.0001)
    assert np.all(independent_aet >= -1e-10)
    assert np.all(independent_aet <= np.minimum(p, kc * et0) + 1e-10)


def test_z_and_awc_cannot_be_identified_separately_from_annual_yield():
    p = np.array([800.0, 300.0])
    args = [p, p * 0.6, np.full(2, 0.65), p * 0.4, p, np.full(2, 0.1), np.ones(2)]
    doubled_awc = [*args[:5], args[5] * 2, args[6]]
    np.testing.assert_allclose(independent_yield(*args, z=10), independent_yield(*doubled_awc, z=5))
