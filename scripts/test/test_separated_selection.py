"""Check shared-parameter gradients and isolation of all post-training information."""

import json

import numpy as np

from ecologyhydro.class_response import evaluate
from ecologyhydro.separated_selection import choose, expansion, fit_past


def sample():
    rng = np.random.default_rng(452)
    zone = np.tile(np.repeat(np.arange(11), 2), 8)
    groups = np.repeat(np.arange(8), 22) * 11 + zone
    n = len(zone)
    return dict(
        zone=zone,
        groups=groups,
        p=rng.uniform(400, 800, n),
        et=rng.uniform(500, 900, n),
        awc=rng.uniform(70, 150, n),
        kc=np.full(n, 0.6),
        veg=np.ones(n),
        weights=np.full(n, 0.3),
        response_group=np.full(n, 3),
    )


def test_grouped_parameter_gradient_matches_finite_difference():
    data = sample()
    for mode in ("global", "regional"):
        matrix, offset = expansion(mode)
        values = np.r_[np.full(matrix.shape[1] - 3, 8.0), 1.1, 15, 0.2]
        _, jac = evaluate(matrix @ values + offset, data, 8, True)
        reduced = jac @ matrix
        for i in range(len(values)):
            delta = np.zeros(len(values))
            delta[i] = 1e-5
            numeric = (
                evaluate(matrix @ (values + delta) + offset, data, 8)
                - evaluate(matrix @ (values - delta) + offset, data, 8)
            ) / 2e-5
            np.testing.assert_allclose(reduced[:, :, i], numeric, rtol=1e-5, atol=1e-6)


def test_future_observations_and_forcing_cannot_change_fit():
    data = sample()
    matrix, offset = expansion("global")
    target = evaluate(matrix @ np.array([8, 1.1, 15, 0.2]) + offset, data, 8)
    adjustment = np.zeros_like(target)
    keep = np.array([False, False, True, True, True, False, False, False])
    first = fit_past({"fine": data}, target, adjustment, np.ones_like(target, bool), keep, "global")
    target[5:], adjustment[5:] = np.nan, np.nan
    data["p"][data["groups"] >= 55] = 1e6
    other = fit_past({"fine": data}, target, adjustment, np.ones_like(target, bool), keep, "global")
    np.testing.assert_array_equal(first["parameters"], other["parameters"])
    assert first["forcing_horizon"] == 5
    assert json.loads(json.dumps(first))["forcing_horizon"] == 5
    assert len(set(first["parameters"][7:12])) == 1


def test_selection_uses_pooled_errors_and_excludes_failed_candidates():
    rows = [
        dict(mode="global", success=True, relative_errors=[0.1, 0.2]),
        dict(mode="regional", success=True, relative_errors=[0.01]),
        dict(mode="regional", success=False, relative_errors=[]),
    ]
    assert choose(rows, ["global", "regional"])["mode"] == "global"
