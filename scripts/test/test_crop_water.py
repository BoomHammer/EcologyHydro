"""Conservation and account-boundary checks for the crop-process diagnostic."""

import numpy as np
import pytest

from ecologyhydro.crop_water import consumptive_supplement, managed_balance, rain_bucket


def test_drought_cannot_create_water_and_carryover_survives_year_boundary():
    p = np.zeros((24, 2))
    p[11] = 90
    d = np.full_like(p, 20)
    result = rain_bucket(p, d, np.array([100, 0]), initial_fraction=0)
    assert result["aet"][12, 0] > 0
    assert result["aet"][12, 1] == 0
    np.testing.assert_allclose(
        p.sum(axis=0),
        result["aet"].sum(axis=0) + result["q"].sum(axis=0) + result["final_soil"],
    )
    np.testing.assert_allclose(result["closure"], 0, atol=1e-12)


def test_rainfed_never_receives_supplement_and_budget_is_capped():
    deficit = np.array([[10, 100], [30, 200]])
    irrigation, used, remainder = consumptive_supplement(deficit, [2, 1], [True, False], 50)
    assert np.all(irrigation[:, 1] == 0)
    assert used == pytest.approx(50)
    assert remainder == pytest.approx(0)
    _, used, remainder = consumptive_supplement(deficit, [2, 1], [True, False], 100)
    assert used == pytest.approx(80)
    assert remainder == pytest.approx(20)
    with pytest.raises(ValueError):
        consumptive_supplement(deficit, [2, 1], [True, False], np.nan)


def test_export_is_not_local_irrigation_and_consumption_is_counted_once():
    # 2024 lower-region account: 100.10 total = 17 local + 83.10 export.
    rain, consumed, export, storage = 150, 100.10, 83.10, -3
    a = managed_balance(rain, consumed, export, 0, storage)
    b = managed_balance(rain, consumed, export, 12, storage)
    assert b["runoff"] == pytest.approx(a["runoff"])
    assert b["residual_consumption"] == pytest.approx(5)
    assert b["runoff"] == pytest.approx(52.90)
    with pytest.raises(ValueError):
        managed_balance(rain, consumed, export, 18, storage)
    missing = managed_balance(rain, np.nan, np.nan, 0, storage)
    assert np.isnan(missing["runoff"])


def test_future_forcing_does_not_change_past_fluxes():
    rng = np.random.default_rng(17)
    p, d = rng.uniform(0, 80, (36, 4)), rng.uniform(0, 100, (36, 4))
    before = rain_bucket(p, d, np.full(4, 120))
    p[24:] *= 10
    after = rain_bucket(p, d, np.full(4, 120))
    np.testing.assert_array_equal(before["q"][:24], after["q"][:24])


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_invalid_forcing_rejected(bad):
    with pytest.raises(ValueError):
        rain_bucket(np.array([[bad]]), np.ones((1, 1)), np.ones(1))
