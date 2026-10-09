"""Check storage signs and equivalence of alternative accounting routes."""

import pytest

from ecologyhydro.water_balance import partial_target


def test_storage_release_can_lower_restored_flow():
    assert partial_target(100, 2, 1, -10, 1, 0) == 92
    assert partial_target(100, 2, 1, 10, 1, 0) == 112


def test_restore_observation_or_correct_model_not_both():
    observed, model, consumption, storage = 100, 150, 20, -5
    target = partial_target(observed, consumption, 10, storage, 0.5, 1)
    correction = 0.5 * consumption + 10 + storage
    assert target == 115
    assert model - target == (model - correction) - observed


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_consumption_rejected(bad):
    with pytest.raises(ValueError):
        partial_target(100, bad, 0, 0, 1, 0)
