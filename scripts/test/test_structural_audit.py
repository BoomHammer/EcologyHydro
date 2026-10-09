"""Distinguish station net increments from nonnegative local natural yield."""

import pytest

from ecologyhydro.structural_audit import reach_budgets


def test_downstream_decrease_is_preserved_and_budgets_telescope():
    rows = reach_budgets(["a", "b", "c"], dict(a=100, b=120, c=150), dict(a=80, b=60, c=90))
    assert rows[1]["observed_net_increment_1e8_m3"] == -20
    assert rows[1]["unexplained_difference_1e8_m3"] == 40
    assert sum(row["local_awy_1e8_m3"] for row in rows) == 150
    assert sum(row["unexplained_difference_1e8_m3"] for row in rows) == 60


def test_missing_station_does_not_become_zero_or_bridge_reaches():
    rows = reach_budgets(
        ["a", "b", "c", "d"],
        dict(a=100, b=120, c=150, d=160),
        dict(a=80, b=None, c=90, d=100),
    )
    assert rows[1]["observed_net_increment_1e8_m3"] is None
    assert rows[2]["observed_net_increment_1e8_m3"] is None
    assert rows[3]["observed_net_increment_1e8_m3"] == 10


def test_reject_decreasing_unadjusted_awy_and_duplicate_stations():
    with pytest.raises(ValueError, match="nondecreasing"):
        reach_budgets(["a", "b"], dict(a=100, b=90), dict(a=80, b=60))
    with pytest.raises(ValueError, match="Duplicate"):
        reach_budgets(["a", "a"], dict(a=100), dict(a=80))
