"""A validation year requires a frozen model, not a bypass of the training guard."""

import json
from types import SimpleNamespace

import pytest

from ecologyhydro.cache import fingerprint
from ecologyhydro.holdout import verify_holdout


def test_frozen_holdout_rejects_changed_year_parameters_and_data(tmp_path):
    config = SimpleNamespace(
        study=SimpleNamespace(calibration_years=[2019, 2020, 2021, 2022], validation_years=[2023])
    )
    index, scenario, data = [tmp_path / f"{name}.json" for name in ("index", "scenario", "data")]
    for path in (index, scenario, data):
        path.write_text("{}")
    frozen = {
        "training_years": [2019, 2020, 2021, 2022],
        "test_year": 2023,
        "parameters": {"fine": {"z": 20}},
        "index_sha256": fingerprint(index)["sha256"],
        "scenarios": {"fine": fingerprint(scenario)},
        "inputs": [fingerprint(data)],
    }
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps(frozen))
    verify_holdout(manifest, config, 2023, "fine", 20, scenario, index)
    for year, z in ((2022, 20), (2023, 21)):
        with pytest.raises(ValueError, match="Holdout year"):
            verify_holdout(manifest, config, year, "fine", z, scenario, index)
    data.write_text("changed")
    with pytest.raises(ValueError, match="input has changed"):
        verify_holdout(manifest, config, 2023, "fine", 20, scenario, index)
