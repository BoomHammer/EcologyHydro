"""Prevent unlabeled, misdated or incomplete precipitation substitutions."""

import json

import numpy as np
import pytest

from ecologyhydro.baseline import require_current_inputs, require_run_inputs
from ecologyhydro.simulation import climate_label, validate_climate_scenario
from ecologyhydro.spatial import write_raster


def test_precipitation_override_requires_year_label_and_coverage(tmp_path):
    zones, rain = tmp_path / "zones.tif", tmp_path / "rain.tif"
    gt = (0, 100, 0, 200, 0, -100)
    write_raster(zones, np.ones((2, 2)), gt, "EPSG:32650")
    write_raster(rain, np.full((2, 2), 600), gt, "EPSG:32650")
    scenario = {
        "overrides": {"precipitation_path": str(rain)},
        "year": 2019,
        "precipitation_label": "provider_v1.1_ppt",
    }
    validate_climate_scenario(scenario, zones, 2019)
    with pytest.raises(ValueError, match="matching year"):
        validate_climate_scenario(scenario, zones, 2020)
    with pytest.raises(ValueError, match="precipitation_label"):
        validate_climate_scenario({**scenario, "precipitation_label": ""}, zones, 2019)
    write_raster(rain, np.array([[600, -9999], [600, 600]]), gt, "EPSG:32650")
    with pytest.raises(ValueError, match="missing or invalid"):
        validate_climate_scenario(scenario, zones, 2019)


def test_both_climate_labels_follow_actual_overrides():
    prepared = {
        "precipitation": "prec",
        "science_scenario": {"precipitation_label": "TC_v1.1_ppt", "et0_label": "TC_v1.1_pet"},
    }
    assert climate_label(prepared) == "TC_v1.1_ppt+TC_v1.1_pet"
    del prepared["science_scenario"]["et0_label"]
    assert climate_label(prepared) == "TC_v1.1_ppt+TerraClimate_PET"


def test_stale_baseline_is_rejected_but_new_log_metadata_is_allowed():
    frozen = {k: f"version1_{k}" for k in ("grid", "recipe", "aligned", "routing", "biophysical")}
    require_current_inputs(frozen, {**frozen, "elapsed_seconds": 9})
    for field in ("aligned", "routing", "biophysical"):
        with pytest.raises(ValueError, match="stale"):
            require_current_inputs(frozen, {**frozen, field: "version2"})
    with pytest.raises(ValueError, match="stale"):
        require_current_inputs({}, {})


def test_old_run_cannot_be_paired_with_new_index(tmp_path):
    keys = {
        "lulc_path": "lulc_fine",
        "precipitation_path": "prec_2019",
        "eto_path": "pet_2019",
        "pawc_path": "pawc",
        "depth_to_root_rest_layer_path": "root_depth",
    }
    index = {
        "routing": {"partitions": str(tmp_path / "basin")},
        "aligned": {v: str(tmp_path / f"{v}.tif") for v in keys.values()},
    }
    prepared = {
        "zone_station": str(tmp_path / "basin/zone_station.csv"),
        "scope": "full",
        "year": 2019,
        "landcover": "fine",
        "precipitation": "prec",
        "args": {},
    }
    for key, name in keys.items():
        folder = tmp_path / key
        folder.mkdir()
        (folder / "manifest.json").write_text(
            json.dumps({"sources": [{"path": index["aligned"][name]}]}), encoding="utf-8"
        )
        prepared["args"][key] = str(folder / "data.tif")
    require_run_inputs(prepared, index)
    index["aligned"]["pawc"] = str(tmp_path / "new_soil.tif")
    with pytest.raises(ValueError, match="different M2 input: pawc"):
        require_run_inputs(prepared, index)
