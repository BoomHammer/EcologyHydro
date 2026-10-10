"""Physical invariants and no-future-target checks for the custom pilot."""

import numpy as np

from ecologyhydro.monthly_pilot import annual_awy, fit_model, simulate


def fixture():
    forcing = dict(
        year=np.repeat([2012, 2013], 12),
        month=np.tile(np.arange(1, 13), 2),
        p=np.full(24, 30.0),
        pet=np.full(24, 20.0),
        temperature=np.full(24, 5.0),
    )
    units = dict(
        awc=np.array([100.0, 0.0]),
        veg=np.array([1, 0]),
        kc=np.ones((24, 2)),
        weights=np.array([0.7, 0.3]),
        volume=1.0,
    )
    return forcing, units


def test_monthly_mass_balance_and_nonnegative_states():
    forcing, units = fixture()
    forcing["temperature"][:5] = -5
    result = simulate([1.0, 1.0, 3.0], forcing, units)
    assert np.max(np.abs(result["closure"])) < 1e-10
    for key in ("q", "aet", "soil", "snow", "reservoir"):
        assert (result[key] >= 0).all()
    assert result["snow"][4] == 150
    assert result["snow"][5] == 0
    assert result["q"][0] == 0


def test_monthly_forcing_is_causal():
    forcing, units = fixture()
    before = simulate([1.0, 1.0, 3.0], forcing, units)
    forcing["p"][12:] *= 10
    after = simulate([1.0, 1.0, 3.0], forcing, units)
    for key in before:
        np.testing.assert_array_equal(before[key][:12], after[key][:12])


def test_future_and_missing_targets_never_enter_fit():
    observed = np.array([2.0, 3.0, np.nan, 999.0])
    adjustment = np.array([0.0, 0.0, np.nan, -999.0])
    keep = np.array([True, True, False, False])

    def predict(x):
        return np.array([x[0], x[0] + x[1], x[0], x[1]])

    first = fit_model(predict, observed, adjustment, keep, "annual_local")
    observed[-1], adjustment[-1] = -123456.0, 123456.0
    second = fit_model(predict, observed, adjustment, keep, "annual_local")
    np.testing.assert_array_equal(first["parameters"], second["parameters"])
    assert first["training_indices"] == [0, 1]


def test_annual_equation_matches_official_kernel():
    from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

    p = np.array([[100.0, 400.0, 1000.0, 500.0]])
    et = np.array([[1000.0, 600.0, 500.0, 900.0]])
    root = np.array([[100.0, 500.0, 800.0, 0.0]])
    pawc = np.full_like(p, 0.2)
    kc = np.array([[0.7, 0.8, 1.0, 0.2]])
    veg = np.array([[1, 1, 1, 0]])
    data = dict(p=p, et=et, awc=root * pawc, kc=kc, veg=veg, weights=np.ones_like(p))
    effective = np.where(veg == 1, np.minimum(kc * 1.1, 1.3), kc)
    fraction = fractp_op(
        effective,
        et,
        p,
        root,
        root,
        pawc,
        veg,
        dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999),
        5.0,
    )
    np.testing.assert_allclose(
        annual_awy([5.0, 1.1], data), ((1 - fraction) * p).sum(axis=1), rtol=1e-6
    )
