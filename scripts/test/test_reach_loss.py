"""Check downstream transport, units, and exclusion of the withheld year."""

import numpy as np
import pytest

from ecologyhydro.reach_loss import apply_reach_loss, fit_reach_loss
from ecologyhydro.regional_calibration import ENDPOINTS, predict
from scripts.test.test_regional_calibration import sample_data


def test_loss_is_once_only_and_only_below_lanzhou():
    flow = np.array([[100, 120, 140, 150, 170, 180]], dtype=float)
    result = apply_reach_loss(flow, 25)
    np.testing.assert_array_equal(result, [[100, 120, 115, 125, 145, 155]])
    np.testing.assert_array_equal(flow, [[100, 120, 140, 150, 170, 180]])
    assert apply_reach_loss(np.ones((1, 6)), 25)[0, 2] == -24
    with pytest.raises(ValueError, match="nonnegative"):
        apply_reach_loss(flow, -1)
    with pytest.raises(ValueError, match="finite"):
        apply_reach_loss(flow, np.nan)


def test_recovery_without_withheld_year_leakage():
    data = sample_data()
    known, loss = [4, 8, 12, 20, 1.2], 2.5
    adjustment = np.ones((4, 6))
    target = apply_reach_loss(predict(known, data)[:, ENDPOINTS] - adjustment, loss)
    target[-1] *= 100
    fitted = fit_reach_loss(data, target, adjustment, keep=[True, True, True, False])
    np.testing.assert_allclose(fitted["parameters"], known, atol=0.005)
    assert fitted["annual_loss_1e8_m3"] == pytest.approx(loss, abs=0.005)
