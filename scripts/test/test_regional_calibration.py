"""Check regional physics against InVEST and water accounting independently."""

import numpy as np
import pytest
from natcap.invest.annual_water_yield.annual_water_yield import fractp_op

from ecologyhydro.regional_calibration import (
    ZONE_REGION,
    ZONE_REGION_FOUR,
    fit,
    predict,
    route,
    yield_depth,
)


def sample_data():
    rng = np.random.default_rng(42)
    n = 440
    return {
        "p": rng.uniform(300, 1200, n),
        "et": rng.uniform(300, 1100, n),
        "kc": rng.uniform(0.3, 0.8, n),
        "veg": np.ones(n),
        "awc": rng.uniform(10, 100, n),
        "weights": np.full(n, 0.01),
        "zone": np.arange(n) % 11,
        "groups": np.arange(n) % 44,
    }


@pytest.mark.parametrize("parameters", [[4, 12, 27, 1.4], [4, 8, 12, 27, 1.4]])
def test_regions_match_official_scalar_calls(parameters):
    data = sample_data()
    data["veg"][::7] = 0
    expected = np.empty(440)
    nodata = dict.fromkeys(("out_nodata", "eto", "precip", "depth_root", "pawc"), -9999)
    region_map = ZONE_REGION_FOUR if len(parameters) == 5 else ZONE_REGION
    for region, z in enumerate(parameters[:-1]):
        mask = region_map[data["zone"]] == region
        kc = np.where(data["veg"] == 1, np.minimum(data["kc"] * 1.4, 1.3), data["kc"])
        fraction = fractp_op(
            kc[mask],
            data["et"][mask],
            data["p"][mask],
            np.full(mask.sum(), 1000),
            np.full(mask.sum(), 1000),
            data["awc"][mask] / 1000,
            data["veg"][mask],
            nodata,
            z,
        )
        expected[mask] = (1 - fraction) * data["p"][mask]
    np.testing.assert_allclose(yield_depth(parameters, data), expected, atol=0.0001)
    np.testing.assert_allclose(predict([12, 1.4], data), predict([12, 12, 12, 1.4], data))


def test_accounting_release_and_downstream_decrease():
    # Incremental natural inputs 100 and 10; second reach consumes 30, releases 5.
    actual = route([[100, 110]], [[2, 30]], [[3, -5]])
    np.testing.assert_allclose(actual, [[95, 80]])
    np.testing.assert_allclose(np.diff(actual), [[10 - 30 + 5]])
    assert route([[1]], [[5]], [[0]])[0, 0] == -4
    with pytest.raises(ValueError, match="Missing/invalid"):
        route([[1]], [[np.nan]], [[0]])


def test_nonvegetated_yield_is_not_controlled_by_z_or_soil():
    data = sample_data()
    data["veg"][:] = 0
    data["kc"][:] = 0.15
    expected = np.maximum(data["p"] - 0.15 * data["et"], 0)
    for parameters in ([1, 0.7], [30, 2], [1, 5, 20, 30, 1.8]):
        np.testing.assert_allclose(yield_depth(parameters, data), expected)
    data["awc"] *= 100
    np.testing.assert_allclose(yield_depth([30, 2], data), expected)


@pytest.mark.parametrize("known", [[4, 12, 20, 1.2], [4, 8, 12, 20, 1.2]])
def test_fitting_excludes_withheld_year_and_recovers_regional_parameters(known):
    from ecologyhydro.regional_calibration import ENDPOINTS

    data = sample_data()
    correction = np.full((4, 6), 1.0)
    target = predict(known, data)[:, ENDPOINTS] - correction
    target[-1] *= 100
    fitted = fit(
        data,
        target,
        correction,
        regional=True,
        keep=[True, True, True, False],
        regions=len(known) - 1,
    )
    np.testing.assert_allclose(fitted["parameters"], known, atol=0.005)


@pytest.mark.parametrize("parameters", [[5, 12, 20, 1.1], [5, 8, 12, 20, 1.1]])
def test_block_sampling_and_full_kernel_preserve_area_and_nested_totals(tmp_path, parameters):
    from osgeo import gdal, osr

    from scripts.calibrate_regional_water import full_year, sample_year, sampling_positions

    gdal.UseExceptions()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(6933)
    zone = np.tile(np.arange(1, 12), (130, 1)).astype(float)
    zone[0, 0] = 0  # Outside catchments must not contribute any water.
    values = {
        "zone": zone,
        "code": np.ones_like(zone),
        "p": np.full_like(zone, 600),
        "et": np.full_like(zone, 800),
        "pawc": np.full_like(zone, 0.12),
        "soil": np.full_like(zone, 750),
    }
    paths = {}
    for key, array in values.items():
        paths[key] = str(tmp_path / f"{key}.tif")
        with gdal.GetDriverByName("GTiff").Create(
            paths[key], 11, 130, 1, gdal.GDT_Float32
        ) as dataset:
            dataset.SetGeoTransform((0, 250, 0, 40000, 0, -250))
            dataset.SetProjection(spatial_ref.ExportToWkt())
            dataset.GetRasterBand(1).SetNoDataValue(-9999)
            dataset.GetRasterBand(1).WriteArray(array)
    table = [{"lucode": 1, "kc": 0.8, "root_depth": 1000, "lulc_veg": 1}]
    positions, weights, volume = sampling_positions(paths, budget=50)
    assert weights.sum() == pytest.approx((zone > 0).sum() * 62500 / 1e11)
    sample = sample_year(paths, table, positions, weights)
    complete = full_year(paths, table, {"regional": parameters}, volume)["regional"]
    np.testing.assert_allclose(predict(parameters, sample, 1)[0], complete, atol=1e-6)
    with gdal.Open(paths["p"], gdal.GA_Update) as dataset:
        dataset.GetRasterBand(1).WriteArray(np.array([[-9999.0]]), xoff=1, yoff=0)
    with pytest.raises(ValueError, match="Invalid coverage"):
        full_year(paths, table, {"regional": parameters}, volume)


def test_structural_error_floor_uses_relative_error_weights():
    from scripts.summarize_regional_water import monotone_lower_bound

    fitted, lower = monotone_lower_bound([100.0, 80.0])
    # Minimizer of ((x-100)/100)^2 + ((x-80)/80)^2 is not the arithmetic mean.
    x = (1 / 100 + 1 / 80) / (1 / 100**2 + 1 / 80**2)
    np.testing.assert_allclose(fitted, [x, x])
    assert lower == pytest.approx(np.sqrt(((x / 100 - 1) ** 2 + (x / 80 - 1) ** 2) / 2) * 100)
    _, zero = monotone_lower_bound([80.0, 100.0])
    assert zero == 0
