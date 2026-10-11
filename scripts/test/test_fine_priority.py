"""Verify response replacement, water conservation and held-year isolation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from test_class_response import fixture_data

from ecologyhydro import fine_priority as fine
from ecologyhydro.class_response import evaluate


def test_head_replacement_identity_and_regional_closure():
    data, _ = fixture_data()
    x = np.r_[[8.0] * 7, [1.2] * 5, 0.2, 10, 0.2]
    expected = evaluate(x, data, 3)
    actual = fine.predict_candidate(data, x, None, [x[0], 1, x[14]], np.ones(3), 3)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    no_storage = x.copy()
    no_storage[14], no_storage[13] = 0, 0
    natural = evaluate(no_storage, data, 3)
    delivered, change = fine.regional_delivery(natural, np.full(7, x[14]))
    delivered[:, 2:] -= x[13]
    np.testing.assert_allclose(delivered, expected, atol=1e-10)
    np.testing.assert_allclose(
        np.diff(natural, axis=1, prepend=0),
        np.diff(delivered + np.array([0, 0, 10, 10, 10, 10, 10]), axis=1, prepend=0) + change,
    )


def test_reference_fill_uses_only_training_years():
    climate = pd.DataFrame(
        dict(
            region=[0] * 4,
            year=[2011, 2013, 2014, 2024],
            p=[100] * 4,
            reference_p=[np.nan, 80, 90, 1000],
        )
    )
    factors = fine.rainfall_factors(
        climate, [2011, 2013, 2014, 2024], np.array([False, True, True, False]), True
    )
    np.testing.assert_allclose(factors, [0.85, 0.8, 0.9, 10])


def test_head_fit_excludes_future_observations_and_forcing(monkeypatch):
    data, _ = fixture_data()
    first = data["groups"] // 11 == 0
    for year in (3, 4):
        extra = {key: value[first].copy() for key, value in data.items()}
        extra["groups"] += year * 11
        data = {key: np.r_[value, extra[key]] for key, value in data.items()}
        first = data["groups"] // 11 == 0
    x = np.r_[[8.0] * 7, [1.2] * 5, 0.2, 10, 0.2]
    observed = evaluate(x, data, 5)
    adjustment = np.zeros_like(observed)
    valid = np.ones_like(observed, bool)
    keep = np.array([True, True, True, True, False])
    settings = dict(
        headwater_bounds=[[1, 0.5, 0], [30, 1.5, 0.5]], headwater_starts=[[8, 1, 0.2]], max_nfev=10
    )

    def fixed(fun, start, **kwargs):
        residual = fun(start)
        return SimpleNamespace(
            x=np.array(start),
            fun=residual,
            success=True,
            cost=float(residual @ residual / 2),
            active_mask=np.zeros(3),
            nfev=1,
        )

    monkeypatch.setattr(fine, "least_squares", fixed)
    before = fine.fit_head(data, observed, adjustment, valid, keep, x, np.ones(5), settings)
    observed[-1] = np.nan
    adjustment[-1] = np.nan
    data["p"][data["groups"] // 11 == 4] = 1e10
    after = fine.fit_head(data, observed, adjustment, valid, keep, x, np.ones(5), settings)
    assert before == after
