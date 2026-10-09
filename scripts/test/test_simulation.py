"""Unit, missing-data, accounting and official-model integration regressions."""

import csv
import json

import numpy as np
import pytest
from osgeo import gdal, ogr

from ecologyhydro.aggregation import (
    official_volume_check,
    read_observations,
    station_totals,
    zone_volumes,
)
from ecologyhydro.cache import fingerprint
from ecologyhydro.model_inputs import mask_input
from ecologyhydro.simulation import valid_completed
from ecologyhydro.spatial import NODATA, crs, write_raster
from ecologyhydro.watersheds import append_feature, make_layer

gdal.UseExceptions()


def raster(path, data):
    write_raster(path, np.asarray(data, dtype=np.float32), (0, 250, 0, 500, 0, -250), "EPSG:32650")
    return path


def zone_raster(path, data):
    values = np.asarray(data, dtype=np.uint8)
    with gdal.GetDriverByName("GTiff").Create(
        str(path), values.shape[1], values.shape[0], 1, gdal.GDT_Byte
    ) as dataset:
        dataset.SetGeoTransform((0, 250, 0, 500, 0, -250))
        dataset.SetProjection(crs("EPSG:32650").ExportToWkt())
        dataset.GetRasterBand(1).SetNoDataValue(0)
        dataset.GetRasterBand(1).WriteArray(values)
    return path


def test_units_and_complete_valid_area(tmp_path):
    zones = zone_raster(tmp_path / "zones.tif", [[1, 1], [2, 0]])
    rain = raster(tmp_path / "rain.tif", [[1000, 1000], [1000, NODATA]])
    aet = raster(tmp_path / "aet.tif", [[900, 800], [700, NODATA]])
    wyield = raster(tmp_path / "yield.tif", [[100, 200], [300, NODATA]])
    rows, checks = zone_volumes(wyield, zones, rain, aet)
    assert [row["volume_m3"] for row in rows] == [18750, 18750]
    assert rows[0]["area_km2"] == 0.125
    assert rows[1]["valid_area_km2"] == 0.0625
    assert checks["maximum_water_balance_error_mm"] == 0


@pytest.mark.parametrize("value", [NODATA, -2, -0.001, np.nan])
def test_missing_or_invalid_yield_is_not_partial_success(tmp_path, value):
    zones = zone_raster(tmp_path / "zones.tif", [[1, 1], [0, 0]])
    wyield = raster(tmp_path / "yield.tif", [[100, value], [NODATA, NODATA]])
    with pytest.raises(ValueError, match="Missing/negative"):
        zone_volumes(wyield, zones)


def test_water_balance_failure(tmp_path):
    zones = zone_raster(tmp_path / "zones.tif", [[1, 1], [0, 0]])
    wyield = raster(tmp_path / "yield.tif", [[100, 100], [NODATA, NODATA]])
    rain = raster(tmp_path / "rain.tif", [[1000, 1000], [NODATA, NODATA]])
    aet = raster(tmp_path / "aet.tif", [[800, 800], [NODATA, NODATA]])
    with pytest.raises(ValueError, match="water-balance"):
        zone_volumes(wyield, zones, rain, aet)


def test_float32_negative_roundoff_is_accounted_without_losing_area(tmp_path):
    zones = zone_raster(tmp_path / "zones.tif", [[1, 1], [0, 0]])
    wyield = raster(tmp_path / "yield.tif", [[100, -0.0001], [NODATA, NODATA]])
    rows, checks = zone_volumes(wyield, zones)
    assert rows[0]["volume_m3"] == pytest.approx(6250)
    assert rows[0]["valid_area_km2"] == rows[0]["area_km2"]
    assert checks["negative_roundoff_cells"] == 1
    assert checks["negative_roundoff_correction_m3"] == pytest.approx(0.00625)


def test_nested_station_totals_and_duplicate_mapping(tmp_path):
    path = tmp_path / "mapping.csv"
    path.write_text("zone_id,station_id,station\n1,1,A\n1,2,B\n2,2,B\n")
    records = [
        {"zone_id": i, "volume_m3": 1e8 * i, "area_km2": i, "valid_area_km2": i} for i in (1, 2)
    ]
    observations = {name: {"raw": "", "value": None, "status": "missing"} for name in ("A", "B")}
    rows = station_totals(records, path, observations, "trial", 2020, "fine", "prec", 2)
    assert [row["natural_yield_1e8_m3"] for row in rows] == [1, 3]
    assert rows[1]["mean_flow_m3_s"] == pytest.approx(3e8 / (366 * 86400))
    assert rows[1]["observed_1e8_m3"] is None
    path.write_text(path.read_text() + "1,2,B\n")
    with pytest.raises(ValueError, match="Duplicate"):
        station_totals(records, path, observations, "trial", 2020, "fine", "prec", 2)


def test_observation_missing_and_invalid_are_distinct(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text("测站,2019\nA,\nB,0\nC,-3\nD,1.2\n", encoding="utf-8")
    observed = read_observations(path, 2019)
    assert observed["A"]["status"] == "missing"
    assert observed["B"]["status"] == "invalid_observation"
    assert observed["C"]["raw"] == "-3"
    assert observed["D"]["value"] == 1.2


def test_mask_undefined_outside_but_reject_undefined_inside(tmp_path):
    zones = zone_raster(tmp_path / "zones.tif", [[1, 1], [0, 0]])
    lulc = raster(tmp_path / "lulc.tif", [[1, 1], [0, 0]])
    output = tmp_path / "masked.tif"
    mask_input(lulc, zones, [0, 0, 2, 2], output, [1])
    with gdal.Open(str(output)) as dataset:
        np.testing.assert_array_equal(dataset.ReadAsArray(), [[1, 1], [NODATA, NODATA]])
    with gdal.Open(str(lulc)) as dataset:
        assert dataset.ReadAsArray()[1, 0] == 0
    raster(lulc, [[0, 1], [0, 0]])
    with pytest.raises(ValueError, match="unmapped"):
        mask_input(lulc, zones, [0, 0, 2, 2], tmp_path / "bad.tif", [1])


def test_completed_run_rejects_modified_or_missing_result(tmp_path):
    result = tmp_path / "stations.csv"
    result.write_text("original result")
    record = {
        "status": "success",
        "run_dir": str(tmp_path),
        "outputs": [{"path": result.name, "sha256": fingerprint(result)["sha256"]}],
    }
    assert valid_completed(record)
    result.write_text("modified result")
    assert not valid_completed(record)
    result.unlink()
    assert not valid_completed(record)
    assert not valid_completed({"status": "success", "run_dir": str(tmp_path), "outputs": []})


def test_official_awy_small_integration(tmp_path):
    from natcap.invest.annual_water_yield import annual_water_yield

    paths = {
        "lulc_path": raster(tmp_path / "lulc.tif", [[1, 2], [1, 2]]),
        "precipitation_path": raster(tmp_path / "rain.tif", np.full((2, 2), 1000)),
        "eto_path": raster(tmp_path / "et.tif", np.full((2, 2), 500)),
        "pawc_path": raster(tmp_path / "pawc.tif", np.full((2, 2), 0.2)),
        "depth_to_root_rest_layer_path": raster(tmp_path / "depth.tif", np.full((2, 2), 1000)),
    }
    table = tmp_path / "bio.csv"
    table.write_text("lucode,lulc_veg,root_depth,kc\n1,1,1000,1\n2,0,1,1\n")
    vector_path = tmp_path / "watersheds.gpkg"
    vector, layer = make_layer(
        vector_path, "basins", "EPSG:32650", ogr.wkbPolygon, [("ws_id", ogr.OFTInteger)]
    )
    append_feature(
        layer, ogr.CreateGeometryFromWkt("POLYGON ((0 0,500 0,500 500,0 500,0 0))"), {"ws_id": 1}
    )
    vector.Close()
    args = {key: str(value) for key, value in paths.items()}
    args.update(
        workspace_dir=str(tmp_path / "model"),
        watersheds_path=str(vector_path),
        biophysical_table_path=str(table),
        seasonality_constant=5,
        n_workers=-1,
    )
    assert not annual_water_yield.validate(args)
    result = annual_water_yield.execute(args)
    zones = zone_raster(tmp_path / "zones.tif", np.ones((2, 2)))
    records, _ = zone_volumes(result["wyield"], zones, paths["precipitation_path"], result["aet"])
    assert official_volume_check(records, result["watershed_results_wyield"])
    with gdal.Open(result["wyield"]) as dataset:
        values = dataset.ReadAsArray()
        np.testing.assert_allclose(values[:, 1], 500, rtol=1e-6)
        expected_vegetated = 1000 * ((1 + 0.5**2.25) ** (1 / 2.25) - 0.5)
        np.testing.assert_allclose(values[:, 0], expected_vegetated, rtol=1e-6)
    with open(result["watershed_results_wyield_csv"], encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 1
    assert json.dumps(records)
