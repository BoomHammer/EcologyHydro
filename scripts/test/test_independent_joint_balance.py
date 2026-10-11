"""Check joint objective gradients and isolation of held observations/forcing."""

import numpy as np
from test_class_response import fixture_data

from ecologyhydro.class_response import evaluate
from ecologyhydro.independent_joint_balance import joint_problem


def test_joint_problem_training_isolation_and_derivative():
    data, _ = fixture_data()
    x = np.r_[[8.0] * 7, [1.1] * 5, 0.2, 10.0, 0.2]
    first = data["groups"] // 11 == 0
    extra = {key: value[first].copy() for key, value in data.items()}
    extra["groups"] += 33
    data = {key: np.r_[value, extra[key]] for key, value in data.items()}
    x[14] = 0.2
    target = evaluate(x, data, 4) * 0.96
    adjustment = np.full_like(target, 0.5)
    eligible = np.ones_like(target, dtype=bool)
    keep = np.array([True, True, True, False])
    settings = dict(local_balance_weight=1.0, class_regularization=0.005)
    compute, metadata = joint_problem(data, target, adjustment, eligible, keep, settings)
    residual, jacobian = compute(x)
    step = 1e-5
    numeric = np.column_stack(
        [
            (compute(x + np.eye(15)[i] * step)[0] - compute(x - np.eye(15)[i] * step)[0])
            / (2 * step)
            for i in range(15)
        ]
    )
    np.testing.assert_allclose(jacobian, numeric, rtol=2e-4, atol=1e-7)
    target[-1] = np.nan
    adjustment[-1] = np.nan
    data["p"][data["groups"] // 11 == 3] *= 10000
    changed, changed_metadata = joint_problem(data, target, adjustment, eligible, keep, settings)
    assert metadata == changed_metadata
    np.testing.assert_array_equal(changed(x)[0], residual)
    np.testing.assert_array_equal(changed(x)[1], jacobian)
