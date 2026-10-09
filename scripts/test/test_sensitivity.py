"""Check root-distribution units, candidate bounds and nodata preservation."""

import math

import numpy as np
import pytest
import yaml
from osgeo import gdal

from ecologyhydro.config import project_root
from ecologyhydro.sensitivity import candidate_rows, root_values, scale_depth
from ecologyhydro.spatial import NODATA, write_raster


def test_distribution_depth_units_and_invalid_bounds():
    depth = root_values({"beta": math.sqrt(0.05)})[1]
    assert depth == pytest.approx(20)  # 95 percent at 2 cm, converted to mm.
    assert root_values({"fraction_top_300mm": 0.95})[1] == pytest.approx(300)
    with pytest.raises(ValueError):
        root_values({"root_mm": [1000, 500, 1500]})


def test_shared_parameter_policy_and_distinct_alpine_roots():
    spec = yaml.safe_load(
        (project_root() / "config/biophysical_candidates.yaml").read_text(encoding="utf-8")
    )
    fine = {r["lucode"]: r for r in candidate_rows(spec, "fine", "central")}
    coarse = {r["lucode"]: r for r in candidate_rows(spec, "copernicus", "central")}
    for key in ("kc", "root_depth", "lulc_veg"):
        assert fine[2][key] == fine[3][key] == coarse[40][key]
    assert fine[18]["root_depth"] == 300
    assert fine[20]["root_depth"] > 300
    for variant in ("kc_low", "kc_high", "root_low", "root_high"):
        for row in candidate_rows(spec, "fine", variant):
            unchanged = "root_depth" if variant.startswith("kc") else "kc"
            assert row[unchanged] == fine[row["lucode"]][unchanged]


def test_soil_depth_scenario_preserves_mask_and_source(tmp_path):
    source = tmp_path / "source.tif"
    target = tmp_path / "target.tif"
    values = np.array([[NODATA, 300, 1500]], dtype=np.float32)
    write_raster(source, values, (0, 250, 0, 250, 0, -250), "EPSG:3857")
    scale_depth(source, target, 2)
    with gdal.Open(str(target)) as dataset:
        np.testing.assert_array_equal(dataset.ReadAsArray(), [[NODATA, 600, 2000]])
    with gdal.Open(str(source)) as dataset:
        np.testing.assert_array_equal(dataset.ReadAsArray(), values)
