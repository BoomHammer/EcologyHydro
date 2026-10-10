"""Keep rough validation assumptions separate from observed training accounts."""

from pathlib import Path

import numpy as np


def test_rough_accounts_use_only_complete_past_cumulative_values(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from scripts import validate_requested_years as requested

    consumption = np.ones((5, 7))
    storage = np.zeros((5, 7))
    storage[:, 0] = [-3, 1, -1, 0, 4]
    storage[1, 2] = np.nan
    adjustment = np.cumsum(consumption + storage, axis=1)

    def past_only(_root, years):
        assert years == [2013, 2014, 2015, 2016, 2017]
        return np.ones((5, 7)), consumption, storage, adjustment, np.isfinite(adjustment)

    monkeypatch.setattr(requested, "load_accounts", past_only)
    result = requested.rough_accounts(Path("unused"))
    assert result[0]["source_years"] == [2013, 2014, 2015, 2016, 2017]
    assert result[2]["source_years"] == [2013, 2015, 2016, 2017]
    # Complete cumulative accounts: [0, 2, 3, 7], not a zero-filled missing value.
    assert result[2]["mean"] == 3
    assert result[2]["minimum"] == 0
    assert result[2]["maximum"] == 7
    assert result[2]["consumption_mean"] + result[2]["reservoir_change_mean"] == 3
    assert np.isnan(adjustment[1, 2:]).all()
