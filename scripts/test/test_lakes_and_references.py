"""Test shared water masks and independent spatial-reference alarms."""

import numpy as np
import pytest
from osgeo import gdal, ogr

from ecologyhydro.lakes import lake_mask, overlay_lakes
from ecologyhydro.reference_checks import discrepancy, precipitation_diagnostics
from ecologyhydro.spatial import write_raster
from ecologyhydro.watersheds import append_feature, make_layer


def test_polygon_overlay_preserves_island_and_outside_for_both_products(tmp_path):
    template = tmp_path / "fine.tif"
    transform = (0, 1000, 0, 5000, 0, -1000)
    write_raster(template, np.full((5, 5), 18), transform, "EPSG:32650")
    source = tmp_path / "lakes.gpkg"
    vector, layer = make_layer(source, "lakes", "EPSG:32650", ogr.wkbPolygon, [])
    polygon = ogr.CreateGeometryFromWkt(
        "POLYGON ((1000 1000,4000 1000,4000 4000,1000 4000,1000 1000),"
        "(2000 2000,2000 3000,3000 3000,3000 2000,2000 2000))"
    )
    append_feature(layer, polygon, {})
    vector.Close()
    mask_dir = tmp_path / "mask"
    mask_dir.mkdir()
    result = lake_mask(source, template, mask_dir)
    assert result["mask_pixels"] == 8
    for product, original_code, water_code in (("fine", 18, 25), ("copernicus", 30, 80)):
        original = tmp_path / f"{product}.tif"
        write_raster(original, np.full((5, 5), original_code), transform, "EPSG:32650")
        output = tmp_path / product
        output.mkdir()
        change = overlay_lakes(original, mask_dir / "mask.tif", water_code, output)
        with gdal.Open(str(output / "data.tif")) as raster:
            actual = raster.ReadAsArray()
        expected = np.full((5, 5), original_code)
        expected[1:4, 1:4] = water_code
        expected[2, 2] = original_code
        np.testing.assert_array_equal(actual, expected)
        assert change["changed_pixels"] == 8
        with gdal.Open(str(original)) as raster:
            assert np.all(raster.ReadAsArray() == original_code)


def test_precipitation_diagnostic_separates_area_from_depth():
    row = {
        "region": "龙羊峡至兰州",
        "year": 2019,
        "variable": "prec",
        "area_km2": 200,
        "volume_1e8_m3": 2,
        "observed_1e8_m3": 1,
        "valid_fraction": 1,
    }
    thresholds = {
        "area_relative_difference": 0.15,
        "precipitation_volume_relative_difference": 0.3,
        "precipitation_depth_relative_difference": 0.25,
    }
    result = precipitation_diagnostics(row, 100, thresholds)
    assert result["area_relative_difference"] == 1
    assert result["volume_relative_difference"] == 1
    assert result["depth_relative_difference"] == 0
    assert result["review_required"]
    assert "precipitation_depth" not in result["review_reasons"]
    row["volume_1e8_m3"] = 4
    assert precipitation_diagnostics(row, 100, thresholds)["depth_relative_difference"] == 1
    # Reference alarms report disagreement without changing either input.
    assert row["area_km2"] == 200
    assert discrepancy(200, 100, 0.15)["review_required"]
    with pytest.raises(ValueError):
        discrepancy(200, 0, 0.15)
