"""Guard closed-domain topology and honest delete-year stability experiments."""

import numpy as np
import pytest

from ecologyhydro.closed_boundary import close_boundary
from ecologyhydro.regional_calibration import ENDPOINTS, predict
from ecologyhydro.stability import fit_stable, stability_metrics, training_subsets
from scripts.test.test_regional_calibration import sample_data


def test_closed_boundary_cuts_exit_preserving_external_and_internal_edges():
    direction = np.array([[0, 0, 0, 128]], dtype=np.uint8)
    inside = np.array([[False, True, True, False]])
    corrected, counts = close_boundary(direction, inside, nodata=128)
    np.testing.assert_array_equal(corrected, [[0, 0, 128, 128]])
    np.testing.assert_array_equal(direction, [[0, 0, 0, 128]])
    assert sum(counts) == 1
    np.testing.assert_array_equal(close_boundary(corrected, inside, 128)[0], corrected)


def test_closed_boundary_handles_diagonal_and_grid_edges():
    direction = np.array([[1, 0], [2, 128]], dtype=np.uint8)
    corrected, counts = close_boundary(direction, np.ones((2, 2), dtype=bool), 128)
    np.testing.assert_array_equal(corrected, [[128, 128], [2, 128]])
    assert sum(counts) == 2
    with pytest.raises(ValueError, match="Invalid D8"):
        close_boundary(direction, np.ones((2, 2)), 0)


def test_nested_subset_members_never_include_outer_withheld_year():
    assert training_subsets([True] * 4) == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for withheld in range(4):
        subsets = training_subsets(np.arange(4) != withheld)
        assert len(subsets) == 3
        assert all(len(s) == 2 and withheld not in s for s in subsets)
    with pytest.raises(ValueError, match="three or four"):
        training_subsets([True, True, False, False])


@pytest.mark.parametrize("regions", [3, 4])
def test_training_fit_is_invariant_to_excluded_observations_and_accounts(regions):
    data = sample_data()
    parameters = [4, 8, 12, 1.2] if regions == 3 else [4, 8, 12, 20, 1.2]
    target = predict(parameters, data)[:, ENDPOINTS]
    adjustment = np.zeros((4, 6))
    keep = [True, True, True, False]
    original = fit_stable(data, target, adjustment, keep, regions)
    target[-1] *= 100
    adjustment[-1] = 99999
    poisoned = fit_stable(data, target, adjustment, keep, regions)
    np.testing.assert_array_equal(original["parameters"], poisoned["parameters"])
    assert original["annual_loss_1e8_m3"] == poisoned["annual_loss_1e8_m3"]


def test_stability_does_not_confuse_low_average_error_with_worst_year():
    target = np.full((4, 6), 100.0)
    predictions = target.copy()
    predictions[2] = 140
    folds = np.broadcast_to(target, (4, 4, 6)).copy()
    folds[0] += 10
    result = stability_metrics(predictions, target, folds, target)
    assert result["mape_pct"] == pytest.approx(10)
    assert result["worst_year_mape_pct"] == pytest.approx(40)
    assert result["delete_year_prediction_rms_change_pct"] == pytest.approx(5)
    assert result["delete_year_prediction_max_change_pct"] == pytest.approx(10)
