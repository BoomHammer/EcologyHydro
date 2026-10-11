"""Check common parameter transfer, seasonal isolation and all twelve tolerances."""

import numpy as np
import pandas as pd
import pytest
from test_class_response import fixture_data

from ecologyhydro.accuracy_goals import PRODUCTS, assess_transfer
from ecologyhydro.calibration_followup import freeze, predict, seasonal_data, seasonality_factors
from ecologyhydro.class_response import evaluate
from ecologyhydro.long_record import STATIONS


def test_common_schema_reproduces_original_and_year_specific_z():
    data, _ = fixture_data()
    seed = np.r_[[8.0] * 7, [1.2] * 5, 0.2, 10, 0.0]
    frozen = freeze(seed, None, {}, np.ones(3), "local")
    np.testing.assert_allclose(predict(data, frozen, 3), evaluate(seed, data, 3), atol=1e-10)
    factors = np.array([0.8, 1, 1.2])
    actual = evaluate(seed, seasonal_data(data, factors), 3)
    for yi, factor in enumerate(factors):
        changed = seed.copy()
        changed[0] *= factor
        np.testing.assert_allclose(actual[yi], evaluate(changed, data, 3)[yi], atol=1e-10)


def test_seasonality_reference_excludes_held_years():
    monthly = np.ones((4, 12))
    keep = np.array([True, True, False, False])
    before = seasonality_factors(monthly, keep)
    monthly[-1] = [12] + [0] * 11
    after = seasonality_factors(monthly, keep)
    np.testing.assert_array_equal(before[:-1], after[:-1])
    assert after[-1] == pytest.approx(1 / 12)
    with pytest.raises(ValueError):
        seasonality_factors(np.zeros((4, 12)), keep)


def transfer_frame():
    return pd.DataFrame(
        [
            dict(
                parameter_source=source,
                product=product,
                year=year,
                station=station,
                predicted=101.0,
                observed=100.0,
                eligible=True,
            )
            for source in PRODUCTS
            for product in PRODUCTS
            for year in (2022, 2023, 2024)
            for station in STATIONS
        ]
    )


def test_transfer_margin_boundary_and_one_failed_source():
    frame = transfer_frame()
    frame.loc[frame["product"] == "fine", "predicted"] = 102
    outcome, matrix, _ = assess_transfer(frame)
    assert outcome["transfer_within_margin"] and matrix.shape == (4, 4)
    frame.loc[
        (frame.parameter_source == "worldcover") & (frame["product"] == "fine"), "predicted"
    ] += 1e-7
    outcome, _, _ = assess_transfer(frame)
    assert not outcome["transfer_within_margin"]
    assert outcome["passed_comparisons"] == 9


@pytest.mark.parametrize("problem", ["missing", "duplicated", "different_source_observations"])
def test_invalid_transfer_is_rejected(problem):
    frame = transfer_frame()
    if problem == "missing":
        frame = frame.iloc[:-1]
    elif problem == "duplicated":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[frame.parameter_source == "worldcover", "observed"] += 1
    with pytest.raises(ValueError):
        assess_transfer(frame)
