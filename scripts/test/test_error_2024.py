"""Selection must reject average gains confined to a subset of products or folds."""

import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.parametrize("failure_axis", [None, "product", "fold"])
def test_selection_requires_distributed_improvement(monkeypatch, failure_axis):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    choose = importlib.import_module("experiment_error_2024").choose
    recipe = {
        "candidates": {"baseline": {}, "alternative": {}},
        "folds": [[2013, 2014, 2015], [2016, 2017], [2019, 2020, 2021]],
        "selection": {"minimum_improved_years": 6, "minimum_improved_stations": 5},
    }
    records = []
    for candidate in recipe["candidates"]:
        for product in range(3):
            for fold, years in enumerate(recipe["folds"]):
                for year in years:
                    for station in range(7):
                        bad = (failure_axis == "product" and product == 0) or (
                            failure_axis == "fold" and fold == 0
                        )
                        error = 10 if candidate == "baseline" else (11 if bad else 1)
                        records.append(
                            dict(
                                candidate=candidate,
                                product=product,
                                fold=fold,
                                year=year,
                                station=station,
                                observed=100,
                                predicted=100 + error,
                            )
                        )
    result = choose(pd.DataFrame(records), recipe)
    assert result["selected"] == ("alternative" if failure_axis is None else "baseline")
    poisoned = pd.DataFrame(records)
    poisoned.loc[0, "year"] = 2024
    with pytest.raises(ValueError, match="calibration folds"):
        choose(poisoned, recipe)
    with pytest.raises(ValueError, match="identical, unique"):
        choose(pd.DataFrame(records).iloc[1:], recipe)
