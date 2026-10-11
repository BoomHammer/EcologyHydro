"""Guard strict small wins, ties, unequal observations and aggregate-only success."""

import pandas as pd
import pytest

from ecologyhydro.accuracy_goals import PRODUCTS, assess
from ecologyhydro.long_record import STATIONS


def fixture():
    return pd.DataFrame(
        [
            dict(product=p, year=y, station=s, observed=100.0, predicted=101.0, eligible=True)
            for p in PRODUCTS
            for y in (2022, 2023, 2024)
            for s in STATIONS
        ]
    )


def test_ties_do_not_win_but_arbitrarily_small_positive_advantage_does():
    frame = fixture()
    assert not assess(frame)[0]["overall_better"]
    frame.loc[frame["product"] == "fine", "predicted"] -= 1e-7
    outcome, _, detail = assess(frame)
    assert outcome["every_item_better"] and outcome["overall_better"]
    assert outcome["winning_pairwise_items"] == 72 and len(detail) == 75


def test_overall_success_does_not_imply_every_item_success():
    frame = fixture()
    mask = frame["product"] == "fine"
    frame.loc[mask, "predicted"] = 100
    frame.loc[frame.index[mask][0], "predicted"] = 102
    outcome, _, _ = assess(frame)
    assert outcome["overall_better"] and not outcome["every_item_better"]


@pytest.mark.parametrize("problem", ["missing", "different_observation", "ineligible"])
def test_invalid_comparison_is_rejected(problem):
    frame = fixture()
    if problem == "missing":
        frame = frame.iloc[:-1]
    elif problem == "different_observation":
        frame.loc[0, "observed"] = 101
    else:
        frame.loc[0, "eligible"] = False
    with pytest.raises(ValueError):
        assess(frame)
