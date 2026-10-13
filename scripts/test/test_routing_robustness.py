"""Verify analytic routing gradients, grouped restrictions and held-data isolation."""

import numpy as np

from ecologyhydro.fine_focus_routing import routing_prediction
from ecologyhydro.routing_robustness import fit, forward


def test_analytic_routing_matches_accounting_and_finite_difference():
    rng = np.random.default_rng(7)
    natural = rng.uniform(10, 100, (8, 7)).cumsum(axis=1)
    x = np.r_[np.linspace(0.05, 0.45, 7), 30]
    prediction, jac = forward(natural, x, True)
    expected = routing_prediction(natural, np.zeros_like(natural), x)[0]
    np.testing.assert_allclose(prediction, expected, atol=1e-10)
    for i in range(8):
        delta = np.zeros(8)
        delta[i] = 1e-5
        numeric = (forward(natural, x + delta) - forward(natural, x - delta)) / 2e-5
        np.testing.assert_allclose(jac[:, :, i], numeric, atol=1e-6, rtol=1e-6)


def test_shared_fit_has_two_parameters_and_never_reads_held_observations():
    rng = np.random.default_rng(12)
    natural = rng.uniform(20, 100, (10, 7)).cumsum(axis=1)
    target = forward(natural, np.r_[np.full(7, 0.2), 10])
    known = np.zeros_like(natural)
    weights = np.array([1] * 7 + [0] * 3)
    prior = np.r_[np.full(7, 0.3), 20]
    spec = dict(sharing="shared", ridge=0.001)
    first = fit(natural, target, known, np.ones_like(natural, bool), weights, prior, spec)
    target[7:] = np.nan
    other = fit(natural, target, known, np.ones_like(natural, bool), weights, prior, spec)
    assert first["n_free"] == 2
    np.testing.assert_array_equal(first["parameters"], other["parameters"])
    assert len(set(first["parameters"][:7])) == 1


def test_penalty_reduces_departure_from_historical_prior():
    rng = np.random.default_rng(32)
    natural = rng.uniform(20, 100, (10, 7)).cumsum(axis=1)
    prior = np.r_[np.full(7, 0.1), 20]
    target = forward(natural, np.r_[np.full(7, 0.4), 60])
    results = [
        fit(
            natural,
            target,
            np.zeros_like(natural),
            np.ones_like(natural, bool),
            np.ones(10),
            prior,
            dict(sharing="shared", ridge=r),
        )
        for r in (0, 1)
    ]
    distance = [np.linalg.norm(np.array(result["parameters"]) - prior) for result in results]
    assert distance[1] < distance[0]
