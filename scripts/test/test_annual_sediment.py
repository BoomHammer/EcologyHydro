"""Check virtual nodata handling and fixed-SDR scaling against official kernels."""

import numpy as np
import pygeoprocessing
from natcap.invest.sdr import sdr
from osgeo import gdal, osr

from ecologyhydro.annual_sediment import multiply_vrt, nan_warp, native_factor, raster_grid


def write_raster(path, data):
    spatial = osr.SpatialReference()
    spatial.ImportFromEPSG(32648)
    with gdal.GetDriverByName("GTiff").Create(
        str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Float32
    ) as target:
        target.SetProjection(spatial.ExportToWkt())
        target.SetGeoTransform((400000, 250, 0, 4000000, 0, -250))
        target.GetRasterBand(1).SetNoDataValue(-1)
        target.GetRasterBand(1).WriteArray(data)


def test_virtual_product_preserves_nodata_and_zero(tmp_path):
    gdal.UseExceptions()
    base = np.array([[2, -1, 0], [8, 5, 7]], dtype=np.float32)
    factor = np.array([[3, 4, 2], [0, -1, 0.5]], dtype=np.float32)
    for name, array in (("base", base), ("factor", factor)):
        write_raster(tmp_path / f"{name}.tif", array)
    with gdal.Open(str(tmp_path / "base.tif")) as template:
        grid = raster_grid(template)
        sources = []
        for name in ("base", "factor"):
            output = tmp_path / f"{name}.vrt"
            nan_warp(tmp_path / f"{name}.tif", output, grid)
            sources.append(output)
        multiply_vrt(sources, tmp_path / "product.vrt", template, "t/(ha year)")
    with gdal.Open(str(tmp_path / "product.vrt")) as product:
        actual = product.ReadAsArray()
        np.testing.assert_allclose(actual, [[6, np.nan, 0], [0, np.nan, 3.5]], equal_nan=True)
        np.testing.assert_array_equal(
            product.GetRasterBand(1).GetMaskBand().ReadAsArray() != 0, np.isfinite(actual)
        )


def test_native_grid_does_not_accumulate_float32_coordinate_rounding(tmp_path):
    lon = np.linspace(95.35, 119.45, 242, dtype=np.float32)
    lat = np.linspace(31.55, 42.05, 106, dtype=np.float32)
    values = np.arange(len(lon) * len(lat), dtype=np.float32).reshape(len(lat), len(lon))
    target = tmp_path / "native.tif"
    native_factor(target, lon, lat, values)
    with gdal.Open(str(target)) as raster:
        x, dx, _, y, _, dy = raster.GetGeoTransform()
        np.testing.assert_allclose(
            [x + dx / 2, x + dx * (len(lon) - 0.5), y + dy / 2, y + dy * (len(lat) - 0.5)],
            [lon[0], lon[-1], lat[-1], lat[0]],
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_array_equal(raster.ReadAsArray(), values[::-1])


def test_r_only_scaling_matches_official_sdr_kernels(tmp_path):
    gdal.UseExceptions()
    shape = (4, 4)
    r = np.arange(16, dtype=np.float32).reshape(shape) * 80
    ratio = np.linspace(0, 2, 16, dtype=np.float32).reshape(shape)
    streams = np.zeros(shape, dtype=np.float32)
    streams[0, 2] = 1
    streams[0, 3] = -1
    inputs = {
        "ls": np.full(shape, 3.2),
        "r_base": r,
        "r_year": r * ratio,
        "k": np.full(shape, 0.03),
        "cp": np.full(shape, 0.2),
        "sdr": np.full(shape, 0.35),
        "stream": streams,
    }
    inputs["cp"][1, 1] = 0
    inputs["k"][3, 1] = -1
    for name, values in inputs.items():
        write_raster(tmp_path / f"{name}.tif", values)
    paths = lambda name: str(tmp_path / f"{name}.tif")  # noqa: E731
    for period in ("base", "year"):
        sdr._calculate_rkls(
            paths("ls"), paths(f"r_{period}"), paths("k"), paths("stream"), paths(f"rkls_{period}")
        )
        pygeoprocessing.raster_map(
            op=sdr.usle_op,
            rasters=[paths(f"rkls_{period}"), paths("cp")],
            target_path=paths(f"usle_{period}"),
        )
        pygeoprocessing.raster_map(
            op=np.multiply,
            rasters=[paths(f"usle_{period}"), paths("sdr")],
            target_path=paths(f"export_{period}"),
        )
    with gdal.Open(paths("export_base")) as base, gdal.Open(paths("export_year")) as annual:
        a, b = base.ReadAsArray(), annual.ReadAsArray()
        valid = a != base.GetRasterBand(1).GetNoDataValue()
        np.testing.assert_array_equal(valid, b != annual.GetRasterBand(1).GetNoDataValue())
        np.testing.assert_allclose(a[valid] * ratio[valid], b[valid], rtol=3e-7, atol=1e-7)
