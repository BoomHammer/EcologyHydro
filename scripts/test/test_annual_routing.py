"""Annual storage conserves water, stays causal and excludes unscored accounts."""

import numpy as np

from ecologyhydro.annual_routing import fit_process, predict, route


def test_storage_mass_balance_and_future_causality():
    natural = np.random.default_rng(10).uniform(1, 80, (14, 7))
    parameters = [10, 0.6, 0.4, 5]
    delivered, change, disconnected = route(natural, parameters)
    np.testing.assert_allclose(natural, delivered + change + disconnected, atol=1e-12)
    altered = natural.copy()
    altered[11:] *= 100
    np.testing.assert_array_equal(route(altered, parameters)[0][:11], delivered[:11])
    # Initial equilibrium delivers the first year's connected input exactly.
    np.testing.assert_allclose(delivered[0], natural[0] - disconnected[0])
    assert np.all(delivered >= 0)


def test_fit_excludes_held_observations_and_missing_bridge_accounts():
    natural = np.random.default_rng(8).uniform(10, 80, (11, 7))
    adjustment = np.zeros_like(natural)
    truth = predict(natural, adjustment, [7, 0.65, 0.2, 4])
    keep = np.array([False, False, True, True, False, True, True, False, True, True, True])
    eligible = np.ones_like(natural, dtype=bool)
    eligible[7] = False
    fit = fit_process(natural, truth, adjustment, eligible, keep, 10, [0, 1, 2, 3])
    poisoned = truth.copy()
    poisoned[~keep] = np.nan
    adjustment[~keep] = np.nan
    other = fit_process(natural, poisoned, adjustment, eligible, keep, 10, [0, 1, 2, 3])
    np.testing.assert_allclose(fit["parameters"], [7, 0.65, 0.2, 4], atol=1e-5)
    np.testing.assert_array_equal(fit["parameters"], other["parameters"])


def test_loss_applied_once_at_correct_reaches_and_storage_continuity():
    natural = np.full((11, 7), 30.0)
    prediction = predict(natural, np.zeros_like(natural), [5, 1, 0, 3])
    np.testing.assert_allclose(prediction[0], [30, 60, 85, 115, 145, 175, 202])
    pulse = natural.copy()
    pulse[7] += 20  # Unscored bridge-year climate must still affect the following year.
    base = route(natural, [0, 1, 0.5, 0])[0]
    routed = route(pulse, [0, 1, 0.5, 0])[0]
    np.testing.assert_allclose(routed[8] - base[8], np.full(7, 5.0))
