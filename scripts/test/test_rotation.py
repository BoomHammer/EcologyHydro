"""Annual crop coefficients must conserve potential ET, not add seasons."""

import pytest

from ecologyhydro.rotation import weighted_kc


def test_weighted_kc_conserves_annual_potential_et():
    weights = [10] * 6 + [30] * 6
    coefficients = [0.5] * 6 + [1.1] * 6
    result = weighted_kc(weights, coefficients)
    assert result == pytest.approx(0.95)
    assert result * sum(weights) == pytest.approx(228)
    assert weighted_kc([1] * 12, [0.7] * 12) == pytest.approx(0.7)


@pytest.mark.parametrize("weights", [[0] * 12, [1] * 11, [float("nan")] * 12, [-1] * 12])
def test_invalid_monthly_weights_rejected(weights):
    with pytest.raises(ValueError):
        weighted_kc(weights, [0.7] * 12)
