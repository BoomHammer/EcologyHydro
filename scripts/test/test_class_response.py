"""Check physical equivalence, gradients, hierarchy and paired evaluation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecologyhydro.class_response import (
    collapse_hierarchy,
    evaluate,
    hierarchy_components,
    prepare_response,
)
from ecologyhydro.paired_comparison import paired_scores
from ecologyhydro.regional_transfer import evaluate as baseline_evaluate
from ecologyhydro.shared_biophysical import load_registry


def fixture_data():
    rng = np.random.default_rng(103)
    size = 11 * 3 * 48
    raw = dict(
        groups=np.repeat(np.arange(33), 48),
        zone=np.repeat(np.tile(np.arange(11), 3), 48),
        code=np.tile(np.arange(1, 49), 33),
        weights=np.full(size, 0.002),
        p=rng.uniform(150, 900, size),
        et=rng.uniform(700, 1300, size),
        soil=rng.uniform(300, 2000, size),
        pawc=rng.uniform(0.08, 0.3, size),
    )
    registry = load_registry(Path("config/shared_biophysical.yaml"))
    return prepare_response(raw, registry, "fine"), registry


def test_response_matches_existing_awy():
    data, _ = fixture_data()
    old = np.r_[[9] * 7, 1.3, 10]
    new = np.r_[[9] * 7, [1.3] * 5, 0.15, 10, 0]
    np.testing.assert_allclose(
        evaluate(new.tolist(), data, 3), baseline_evaluate(old, data, "reach7", 3)
    )


def test_all_derivatives():
    data, _ = fixture_data()
    x = np.r_[np.arange(4, 11), [1.1, 1.25, 1.3, 1.6, 0.95], 0.27, 9, 0.2]
    _, analytical = evaluate(x, data, 3, jacobian=True)
    for i in range(len(x)):
        step = np.eye(len(x))[i] * 1e-5
        numerical = (evaluate(x + step, data, 3) - evaluate(x - step, data, 3)) / 2e-5
        np.testing.assert_allclose(analytical[:, :, i], numerical, atol=1e-7, rtol=2e-4)


def test_coarse_removes_subclass_location_but_preserves_area():
    data, registry = fixture_data()
    legend = Path("data/LCLU/vegetation_model_20261016_161145.csv")
    labels, components = hierarchy_components(data, registry, legend)
    coarse = collapse_hierarchy(data, labels, components)
    for parent in set(labels.values()):
        assert sum(c["fraction"] for c in components if c["parent"] == parent) == pytest.approx(1)
    for group in range(33):
        assert coarse["weights"][coarse["groups"] == group].sum() == pytest.approx(
            data["weights"][data["groups"] == group].sum()
        )
    swapped = {key: value.copy() for key, value in data.items()}
    swapped["code"] = np.where(
        data["code"] == 18, 19, np.where(data["code"] == 19, 18, data["code"])
    )
    second = collapse_hierarchy(swapped, labels, components)
    x = np.r_[[9] * 7, [1.3] * 5, 0.15, 10, 0.1]
    np.testing.assert_allclose(evaluate(x, coarse, 3), evaluate(x, second, 3))


def test_paired_comparison_rejects_different_observations():
    rows = [
        dict(product=p, year=2022, station="a", predicted=v, observed=100, eligible=True)
        for p, v in [("fine", 100), ("copernicus", 110), ("worldcover", 120), ("fine_level1", 130)]
    ]
    frame = pd.DataFrame(rows)
    scores, _ = paired_scores(frame)
    assert all(score["wins"] == 1 for score in scores)
    frame.loc[1, "observed"] = 101
    with pytest.raises(ValueError):
        paired_scores(frame)


@pytest.mark.parametrize("mode", ["class", "class_local", "class_storage_robust"])
def test_objective_derivatives_and_excluded_future(monkeypatch, mode):
    import ecologyhydro.class_response as module

    data, _ = fixture_data()
    target = np.full((3, 7), 40.0)
    adjustment = np.ones_like(target)
    keep = np.array([True, True, False])
    datasets = {p: data for p in ("fine", "fine_level1", "copernicus", "worldcover")}

    def capture(fun, x, jac, **_):
        analytical = jac(x)
        for i in range(len(x)):
            step = np.eye(len(x))[i] * 1e-5
            numerical = (fun(x + step) - fun(x - step)) / 2e-5
            np.testing.assert_allclose(analytical[:, i], numerical, atol=2e-8, rtol=1e-3)
        residual = fun(x)
        return SimpleNamespace(
            success=True,
            fun=residual,
            x=x,
            cost=sum(residual**2) / 2,
            nfev=1,
            active_mask=np.zeros(len(x), dtype=int),
        )

    monkeypatch.setattr(module, "least_squares", capture)
    seed = np.r_[[9] * 7, 1.1, 10]
    first = module.fit_shared(datasets, target, adjustment, target > 0, keep, mode, seed)
    eligible = target > 0
    target[2] = np.nan
    data["p"][data["groups"] // 11 == 2] = 1e20
    second = module.fit_shared(datasets, target, adjustment, eligible, keep, mode, seed)
    assert first == second
