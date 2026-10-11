"""Check revised annual targets and a denominator of exactly 21 station-years."""

from test_accuracy_goals import fixture

from ecologyhydro.fine_focus_goals import assess_focus


def test_nine_station_year_wins_are_required_and_annual_columns_are_excluded():
    frame = fixture()
    indices = frame.index[frame["product"] == "fine"]
    frame.loc[indices[:8], "predicted"] = 100.99
    result, _, _ = assess_focus(frame)
    assert result["station_year_wins"] == 8
    assert result["station_year_total"] == 21
    assert not result["station_year_40_percent_best"]
    frame.loc[indices[8], "predicted"] = 100.99
    result, _, _ = assess_focus(frame)
    assert result["station_year_40_percent_best"]
    assert not result["aggregate_all_best"]


def test_first_table_requires_each_annual_and_overall_column():
    frame = fixture()
    frame.loc[frame["product"] == "fine", "predicted"] = 100.9
    assert assess_focus(frame)[0]["aggregate_all_best"]
    frame.loc[(frame["product"] == "fine") & (frame.year == 2023), "predicted"] = 101.1
    result, _, _ = assess_focus(frame)
    assert not result["aggregate_all_best"]
    assert result["station_year_40_percent_best"]
