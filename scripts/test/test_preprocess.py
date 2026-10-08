"""Regression tests for units, masks, cache dependencies and river topology."""

import json
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset, date2num
from osgeo import gdal

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.climate import annual_pet, annual_precipitation, crop_netcdf
from ecologyhydro.conditioning import loop_erased_path
from ecologyhydro.quality import coverage
from ecologyhydro.resampling import compare_resampling
from ecologyhydro.spatial import NODATA, write_raster
from ecologyhydro.watersheds import (
    append_feature,
    mainstem_chain,
    make_layer,
    nearest_on_line,
    partition,
)

gdal.UseExceptions()


def climate_file(path, year=2020, months=12):
    from datetime import datetime

    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", months)
        dataset.createDimension("lat", 5)
        dataset.createDimension("lon", 7)
        t = dataset.createVariable("time", "f8", ("time",))
        t.units = "days since 2000-01-01"
        t.calendar = "standard"
        t[:] = date2num([datetime(year, m, 15) for m in range(1, months + 1)], t.units)
        dataset.createVariable("lat", "f8", ("lat",))[:] = np.arange(30, 35)
        dataset.createVariable("lon", "f8", ("lon",))[:] = np.arange(100, 107)
        data = dataset.createVariable("prec", "f4", ("time", "lat", "lon"), fill_value=-9999)
        data.units = "kg m-2 s-1"
        data[:] = 1 / 86400  # 1 mm/day
        data[0, 1, 1] = -9999


def test_annual_precipitation_calendar_and_missing(tmp_path):
    source, target = tmp_path / "source.nc", tmp_path / "annual.tif"
    climate_file(source)
    annual_precipitation(source, "prec", 2020, target)
    with gdal.Open(str(target)) as dataset:
        values = dataset.ReadAsArray()
        assert values[0, 0] == pytest.approx(366)
        assert values[3, 1] == NODATA  # ascending latitude was flipped
        assert dataset.GetGeoTransform() == (99.5, 1, 0, 34.5, 0, -1)


def test_missing_month_rejected(tmp_path):
    source = tmp_path / "source.nc"
    climate_file(source, months=11)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        annual_precipitation(source, "prec", 2020, tmp_path / "annual.tif")


def test_precipitation_units_rejected(tmp_path):
    source = tmp_path / "source.nc"
    climate_file(source)
    with Dataset(source, "a") as dataset:
        dataset["prec"].units = "mm/month"
    with pytest.raises(ValueError, match="units"):
        annual_precipitation(source, "prec", 2020, tmp_path / "annual.tif")


def test_netcdf_subset_preserves_values_dates_and_mask(tmp_path):
    source, target = tmp_path / "source.nc", tmp_path / "subset.nc"
    climate_file(source)
    crop_netcdf(source, target, [102.5, 31.5, 103.5, 32.5])
    with Dataset(source) as original, Dataset(target) as subset:
        np.testing.assert_array_equal(original["time"][:], subset["time"][:])
        x = np.searchsorted(original["lon"][:], subset["lon"][:])
        y = np.searchsorted(original["lat"][:], subset["lat"][:])
        expected = original["prec"][:, y, x]
        np.testing.assert_array_equal(expected.data, subset["prec"][:].data)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(expected), np.ma.getmaskarray(subset["prec"][:])
        )


def test_pet_scale_no_calendar_multiplier_and_mask(tmp_path):
    sources = []
    for month in range(12):
        path = tmp_path / f"{month}.tif"
        values = np.array([[100, 0], [100, 100]], dtype=np.float32)
        if month == 4:
            values[1, 0] = NODATA
        write_raster(path, values, (100, 1, 0, 35, 0, -1), "EPSG:4326")
        sources.append(path)
    target = tmp_path / "annual.tif"
    annual_pet(sources, target)
    with gdal.Open(str(target)) as dataset:
        np.testing.assert_array_equal(dataset.ReadAsArray(), [[120, 0], [NODATA, 120]])


def test_cache_dependency_isolation_and_failure(tmp_path):
    soil, climate = tmp_path / "soil", tmp_path / "climate"
    soil.write_text("soil")
    climate.write_text("old climate")
    cache = ArtifactCache(tmp_path / "cache")

    def produce(out):
        (out / "value").write_text("complete")

    first = cache.build("static", [soil], {"resolution": 250}, produce)
    weather = cache.build("climate", [climate], {}, produce)
    climate.write_text("new climate with different length")
    assert cache.build("static", [soil], {"resolution": 250}, produce) == first
    assert cache.build("climate", [climate], {}, produce) != weather
    assert cache.build("static", [soil], {"resolution": 100}, produce) != first

    def fail(out):
        (out / "value").write_text("incomplete")
        raise ValueError("interrupted")

    with pytest.raises(ValueError, match="interrupted"):
        cache.build("failed", [soil], {}, fail)
    assert not list((tmp_path / "cache/failed").glob("*/manifest.json"))


def test_cache_detects_modified_output(tmp_path):
    source = tmp_path / "source"
    source.write_text("source")
    cache = ArtifactCache(tmp_path / "cache")

    def produce(out):
        (out / "value").write_text("complete")

    path = cache.build("artifact", [source], {}, produce)
    (path / "value").write_text("changed output")
    with pytest.raises(RuntimeError, match="Modified cache"):
        cache.build("artifact", [source], {}, produce)


def test_mainstem_excludes_nearby_tributary_and_endorheic():
    def row(down, main, area, endorheic=0):
        return {"NEXT_DOWN": down, "MAIN_RIV": main, "UPLAND_SKM": area, "ENDORHEIC": endorheic}

    records = {
        1: row(0, 1, 100),
        2: row(1, 1, 80),
        3: row(1, 1, 10),
        4: row(2, 1, 50),
        5: row(0, 5, 1000, 1),
    }
    assert mainstem_chain(records) == [1, 2, 4]


def test_nearest_point_is_on_segment_not_vertex():
    distance, point = nearest_on_line(np.array([4.0, 2.0]), [(0, 0), (10, 0)])
    assert distance == 2
    np.testing.assert_array_equal(point, [4, 0])


def test_loop_erasure_preserves_neighbor_steps():
    cells = [(0, 0), (1, 0), (1, 1), (1, 0), (2, 0), (3, 1)]
    result = loop_erased_path(cells)
    assert result == [(0, 0), (1, 0), (2, 0), (3, 1)]
    assert np.abs(np.diff(result, axis=0)).max() == 1


def test_manifest_records_provenance(tmp_path):
    source = tmp_path / "source"
    source.write_text("source")
    cache = ArtifactCache(tmp_path / "cache")

    def produce(out):
        (out / "value").write_text("complete")
        return {"units": "mm/year"}

    output = cache.build("artifact", [source], {"scale": 0.1}, produce)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["sources"][0]["sha256"]) == 64
    assert manifest["parameters"]["scale"] == 0.1
    assert manifest["details"]["units"] == "mm/year"
    assert Path(manifest["outputs"][0]["name"]).name == "value"


@pytest.mark.parametrize("nested", [True, False])
def test_partitions_do_not_double_count_or_publish_disconnected_basins(tmp_path, nested):
    from osgeo import ogr

    template = tmp_path / "template.tif"
    write_raster(template, np.ones((2, 4)), (0, 1000, 0, 2000, 0, -1000), "EPSG:32650")
    source = tmp_path / "basins.gpkg"
    vector, layer = make_layer(
        source, "basins", "EPSG:32650", ogr.wkbPolygon, [("ws_id", ogr.OFTInteger)]
    )
    left = "POLYGON ((0 0,2000 0,2000 2000,0 2000,0 0))"
    right = "POLYGON ((2000 0,4000 0,4000 2000,2000 2000,2000 0))"
    whole = "POLYGON ((0 0,4000 0,4000 2000,0 2000,0 0))"
    append_feature(layer, ogr.CreateGeometryFromWkt(left), {"ws_id": 1})
    append_feature(layer, ogr.CreateGeometryFromWkt(whole if nested else right), {"ws_id": 2})
    vector.Close()
    stations = [
        {"ws_id": 1, "station": "upstream", "reference_km2": 999},
        {"ws_id": 2, "station": "downstream", "reference_km2": 9999},
    ]
    output = tmp_path / "output"
    output.mkdir()
    result = partition(source, template, stations, output)
    assert result["accepted"] is nested
    if nested:
        assert result["zone_pixels"] == [0, 4, 4]
        assert result["areas"][1]["area_km2"] == 8
        # Reference area is deliberately wrong: it must not alter delineated geometry.
        with gdal.Open(str(output / "zones.tif")) as zones:
            np.testing.assert_array_equal(zones.ReadAsArray(), [[1, 1, 2, 2], [1, 1, 2, 2]])
    else:
        assert not (output / "zones.tif").exists()


def test_unknown_class_cannot_silently_enter_common_mask(tmp_path):
    transform = (0, 1000, 0, 5000, 0, -1000)
    dem, lulc, scope = (tmp_path / name for name in ("dem.tif", "lulc.tif", "scope.tif"))
    write_raster(dem, np.ones((5, 5)), transform, "EPSG:32650")
    classes = np.ones((5, 5))
    classes[2, 2] = 0
    write_raster(lulc, classes, transform, "EPSG:32650")
    region = np.zeros((5, 5))
    region[1:4, 1:4] = 1
    write_raster(scope, region, transform, "EPSG:32650", nodata=0)
    table = tmp_path / "classes.csv"
    table.write_text("code,name\n1,grass\n")
    result = coverage({"dem": dem, "lulc_fine": lulc}, scope, {"fine": table}, tmp_path)
    assert result["common_pixels"] == 8
    assert result["scope_pixels"] == 9
    assert result["rasters"]["lulc_fine"]["undefined_codes"] == [0]
    assert result["dem_edge_touch_pixels"] == 0


def test_native_and_aligned_constant_rainfall_conserve_volume(tmp_path):
    transform = (0, 1000, 0, 2000, 0, -1000)
    rain, scope = tmp_path / "rain.tif", tmp_path / "scope.tif"
    write_raster(rain, np.full((2, 2), 1000), transform, "EPSG:32650")
    write_raster(scope, np.array([[0, 1], [1, 1]]), transform, "EPSG:32650", nodata=0)
    report = {"native": {"prec_2020": str(rain)}, "aligned": {"prec_2020": str(rain)}}
    result = compare_resampling(report, scope, tmp_path)
    assert result["maximum_absolute_relative_volume_change"] == pytest.approx(0)
