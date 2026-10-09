"""Reject invalid climate overrides and verify the pixelwise yield bound."""

import numpy as np
import pytest
from osgeo import gdal

from ecologyhydro.et0_trials import yield_floor
from ecologyhydro.simulation import validate_et_override
from ecologyhydro.spatial import write_raster


def test_override_masks_and_grid(tmp_path):
    gdal.UseExceptions()
    gt = (0, 10, 0, 20, 0, -10)
    zones, et = tmp_path / "zones.tif", tmp_path / "et.tif"
    write_raster(zones, np.array([[1, 0], [1, 0]]), gt, "EPSG:3857", nodata=0)
    write_raster(et, np.array([[100, -9999], [200, -9999]]), gt, "EPSG:3857")
    validate_et_override(et, zones)
    for bad in (-9999, -1, np.nan):
        write_raster(et, np.array([[bad, -9999], [200, -9999]]), gt, "EPSG:3857")
        with pytest.raises(ValueError, match="missing or invalid"):
            validate_et_override(et, zones)
    write_raster(et, np.ones((2, 2)), (1, 10, 0, 20, 0, -10), "EPSG:3857")
    with pytest.raises(ValueError, match="grid mismatch"):
        validate_et_override(et, zones)


def test_bound_clamps_each_pixel_before_sum(tmp_path):
    gt = (0, 10, 0, 20, 0, -10)
    paths = {}
    for name, array in {
        "zones": [[1, 1]],
        "precipitation_path": [[100, 50]],
        "eto_path": [[200, 200]],
        "lulc_path": [[1, 2]],
    }.items():
        paths[name] = tmp_path / f"{name}.tif"
        write_raster(paths[name], np.array(array), gt, "EPSG:3857")
    table = tmp_path / "table.csv"
    table.write_text("lucode,kc\n1,0.25\n2,1\n", encoding="utf-8")
    result = yield_floor({**paths, "biophysical_table_path": table}, paths["zones"])
    # max(100-50,0)+max(50-200,0)=50 mm on 100 m2 pixels => 5 m3.
    assert result["lower_bound_1e8_m3"] == pytest.approx(5e-8)
    assert result["et0_mean_mm"] == 200
