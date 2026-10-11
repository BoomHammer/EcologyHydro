"""Check analytic carryover derivatives and exclusion of held observations."""

import numpy as np

from ecologyhydro.annual_routing import predict
from ecologyhydro.annual_storage_calibration import evaluate_storage, fit_storage
from ecologyhydro.regional_transfer import evaluate
from scripts.test.test_regional_calibration import sample_data


def test_joint_derivatives_and_independent_mass_balance_model():
    data = sample_data()
    data["veg"][::5] = 0
    x = np.r_[np.linspace(3, 20, 7), 1.8, 2.5, 0.3]
    result, jac = evaluate_storage(x, data, 4, jacobian=True)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = 1e-5
        numeric = (
            evaluate_storage(x + delta, data, 4) - evaluate_storage(x - delta, data, 4)
        ) / 2e-5
        np.testing.assert_allclose(jac[:, :, i], numeric, atol=1e-7, rtol=2e-5)
    awy = x[:9].copy()
    awy[-1] = 0
    natural = evaluate(awy, data, "reach7", 4)
    expected = predict(
        np.diff(natural, axis=1, prepend=0), np.zeros_like(natural), [2.5, 1, 0.3, 0]
    )
    np.testing.assert_allclose(result, expected)
    x[-1] = 0
    np.testing.assert_allclose(evaluate_storage(x, data, 4), evaluate(x[:9], data, "reach7", 4))


def test_joint_fit_does_not_use_held_observations_or_accounts():
    data = sample_data()
    seed = [4, 5, 6, 7, 8, 9, 10, 1.2, 1]
    truth = evaluate_storage([*seed, 0.2], data, 4)
    adjustment = np.zeros_like(truth)
    eligible = np.ones_like(truth, dtype=bool)
    keep = np.array([True, False, True, True])
    fit = fit_storage(data, truth, adjustment, eligible, keep, seed)
    truth[1] = np.nan
    adjustment[1] = np.nan
    changed = fit_storage(data, truth, adjustment, eligible, keep, seed)
    np.testing.assert_array_equal(fit["parameters"], changed["parameters"])
