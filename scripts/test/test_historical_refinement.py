"""Verify isolation, equal map weighting and the spatial penalty derivative."""

import numpy as np
import pytest
from test_separated_selection import sample

from ecologyhydro.class_response import evaluate
from ecologyhydro.historical_refinement import choose_history, fit_history, spatial_penalty
from ecologyhydro.separated_selection import expansion


def test_spatial_penalty_derivative_and_scale_invariance():
    x = np.r_[np.arange(3, 10), 1.1, 15, 0.2]
    residual, jac = spatial_penalty(x, 0.01)
    for i in range(10):
        delta = np.eye(10)[i] * 1e-5
        numeric = (spatial_penalty(x + delta, 0.01)[0] - spatial_penalty(x - delta, 0.01)[0]) / 2e-5
        np.testing.assert_allclose(jac[:, i], numeric, atol=1e-10)
    scaled = x.copy()
    scaled[:7] *= 2
    np.testing.assert_allclose(residual, spatial_penalty(scaled, 0.01)[0], atol=1e-15)


def test_refined_fit_ignores_future_and_repeated_maps_do_not_change_weight():
    data = sample()
    matrix, offset = expansion("regional")
    target = evaluate(matrix @ np.r_[np.arange(5, 12), 1.1, 15, 0.2] + offset, data, 8)
    adjustment = np.zeros_like(target)
    eligible = np.ones_like(target, bool)
    keep = np.array([False, False, True, True, True, False, False, False])
    spec = dict(ridge=0.001, half_life_years=3)
    first = fit_history({"fine": data}, target, adjustment, eligible, keep, spec)
    target[5:], adjustment[5:] = np.nan, np.nan
    data["p"][data["groups"] >= 55] = 1e6
    other = fit_history({"fine": data}, target, adjustment, eligible, keep, spec)
    np.testing.assert_array_equal(first["parameters"], other["parameters"])
    duplicate = fit_history({"a": data, "b": data}, target, adjustment, eligible, keep, spec)
    np.testing.assert_allclose(first["parameters"], duplicate["parameters"], rtol=1e-6, atol=1e-6)
    assert first["forcing_horizon"] == 5
    assert first["unique_observations"] == duplicate["unique_observations"]
    assert first["training_year_weights"][-1] == 1
    assert len(set(first["parameters"][7:12])) == 1


def test_selection_uses_all_maps_and_handles_failures_and_ties():
    rows = [
        dict(mode="baseline", source="fine", success=True, relative_errors=[0.1]),
        dict(mode="baseline", source="other", success=True, relative_errors=[0.1]),
        dict(mode="refined", source="fine", success=True, relative_errors=[0.0]),
        dict(mode="refined", source="other", success=True, relative_errors=[0.3]),
    ]
    assert choose_history(rows, ["baseline", "refined"])["mode"] == "baseline"
    rows[-1]["relative_errors"] = [0.0]
    assert choose_history(rows, ["baseline", "refined"])["mode"] == "refined"
    rows[-1]["success"] = False
    assert choose_history(rows, ["baseline", "refined"])["mode"] == "baseline"
    with pytest.raises(ValueError, match="No converged"):
        choose_history(rows, ["refined"])
