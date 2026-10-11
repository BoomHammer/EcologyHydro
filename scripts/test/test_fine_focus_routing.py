"""Conservation, held-observation isolation and target ranking regressions."""

import numpy as np
import pandas as pd
import pytest

from ecologyhydro.fine_focus_routing import (
    balance_ledger,
    fit_routing,
    routing_prediction,
    selection_key,
)


def test_loss_is_one_local_sink_and_storage_closes_across_years():
    local = np.array([[100] * 7, [200] * 7, [40] * 7], dtype=float)
    natural = local.cumsum(axis=1)
    consumed = np.full_like(natural, 2)
    reservoir = np.full_like(natural, -1)
    parameters = np.r_[np.linspace(0, 0.5, 7), 20]
    result = balance_ledger(natural, natural, consumed, reservoir, parameters)
    np.testing.assert_allclose(result["closure"], 0, atol=1e-10)
    np.testing.assert_allclose(result["loss"].sum(axis=1), 20)
    np.testing.assert_allclose(result["loss"][:, 2], 20)
    initial = local[0] * parameters[:7] / (1 - parameters[:7])
    np.testing.assert_allclose(
        result["storage"][-1] - initial, result["storage_change"].sum(axis=0)
    )
    np.testing.assert_allclose(
        np.diff(result["delivered"], axis=1, prepend=0) + result["storage_change"], local
    )


def test_zero_weight_observations_do_not_affect_fit():
    natural = (np.arange(5)[:, None] * 20 + np.full((5, 7), 100)).cumsum(axis=1)
    initial = np.r_[np.full(7, 0.2), 10]
    adjustment = np.zeros_like(natural)
    target = routing_prediction(natural, adjustment, np.r_[np.full(7, 0.2), 30])[0]
    weights = np.array([1, 1, 0, 1, 0])
    eligible = np.ones_like(natural, bool)
    fit = fit_routing(natural, target, adjustment, eligible, weights, initial, [7])
    target[weights == 0] = np.nan
    other = fit_routing(natural, target, adjustment, eligible, weights, initial, [7])
    np.testing.assert_allclose(fit["parameters"], other["parameters"], atol=1e-12)
    assert fit["parameters"][7] == pytest.approx(30, abs=1e-5)


def test_ranking_enforces_both_goals_before_focus_error():
    passed = dict(
        aggregate_total=4, aggregate_best_count=4, station_year_required=9, station_year_wins=9
    )
    failed = {**passed, "station_year_wins": 8}
    scores = pd.DataFrame({"fine": [6, 5]}, index=["2023_总误差", "三年整体"])
    better_error = scores / 2
    assert selection_key(passed, scores) < selection_key(failed, better_error)
    assert selection_key(passed, better_error) < selection_key(passed, scores)


@pytest.mark.parametrize("bad", [np.nan, -1, 101])
def test_invalid_loss_rejected(bad):
    natural = np.ones((2, 7)).cumsum(axis=1)
    with pytest.raises(ValueError):
        routing_prediction(natural, natural, np.r_[np.zeros(7), bad])
