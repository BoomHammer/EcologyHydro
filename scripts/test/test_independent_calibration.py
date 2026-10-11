"""Check training isolation and the integrity of the four-row deliverable."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_class_response import fixture_data

from ecologyhydro import independent_calibration as independent
from ecologyhydro.class_response import evaluate
from ecologyhydro.long_record import STATIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_independent_calibration import LABELS, main_table  # noqa: E402


@pytest.mark.parametrize("mode", ["class", "class_storage"])
def test_independent_fit_excludes_future_observations_and_forcing(monkeypatch, mode):
    data, _ = fixture_data()
    first = data["groups"] // 11 == 0
    extension = {key: values[first].copy() for key, values in data.items()}
    extension["groups"] += 33
    data = {key: np.r_[values, extension[key]] for key, values in data.items()}
    keep = np.array([True, True, True, False])
    seed = np.r_[[9] * 7, 1.2, 10]
    x = independent.expand_regional(seed)
    target = evaluate(x, data, 4)
    adjustment = np.zeros_like(target)
    eligible = np.ones_like(target, dtype=bool)
    settings = dict(class_regularization=0.005, max_nfev=150)

    def fixed_optimizer(fun, start, jac, **kwargs):
        residual = fun(start)
        assert np.isfinite(jac(start)).all()
        return SimpleNamespace(
            x=start,
            success=True,
            fun=residual,
            cost=float(residual @ residual / 2),
            nfev=1,
            active_mask=np.zeros(len(start), int),
            optimality=0.0,
            message="test",
        )

    monkeypatch.setattr(independent, "least_squares", fixed_optimizer)
    before = independent.fit_independent(
        data, target, adjustment, eligible, keep, mode, seed, settings
    )
    target[-1] *= 10000
    adjustment[-1] = np.nan
    data["p"][data["groups"] // 11 == 3] *= 10000
    after = independent.fit_independent(
        data, target, adjustment, eligible, keep, mode, seed, settings
    )
    assert before == after
    assert before["training_indices"] == [0, 1, 2]


def table_fixture():
    return pd.DataFrame(
        [
            dict(product=p, year=y, station=s, observed=100, predicted=100 + i * 10, eligible=True)
            for i, p in enumerate(LABELS)
            for y in [2022, 2023, 2024]
            for s in STATIONS
        ]
    )


def test_main_table_has_four_rows_and_all_year_station_columns():
    table = main_table(table_fixture(), [2022, 2023, 2024])
    assert table.shape == (4, 24)
    np.testing.assert_array_equal(table.iloc[:, 7], [0, 10, 20, 30])


@pytest.mark.parametrize("problem", ["missing", "duplicate", "changed_observation", "invalid"])
def test_main_table_rejects_incomparable_cells(problem):
    frame = table_fixture()
    if problem == "missing":
        frame = frame.iloc[:-1]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "changed_observation":
        frame.loc[0, "observed"] = 101
    else:
        frame.loc[0, "predicted"] = np.nan
    with pytest.raises(ValueError):
        main_table(frame, [2022, 2023, 2024])
