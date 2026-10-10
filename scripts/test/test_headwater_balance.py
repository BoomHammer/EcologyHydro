"""Verify monthly aggregation against independent constant and linear fields."""

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from ecologyhydro.spatial import ALBERS, crs


@pytest.fixture
def diagnostic(monkeypatch):
    # Command-line scripts resolve sibling helpers from the scripts directory.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from scripts import diagnose_headwater_balance

    return diagnose_headwater_balance


def test_all_pixel_weights_preserve_area_and_linear_field(tmp_path, diagnostic):
    path = tmp_path / "zones.tif"
    grid = (0, 1000, 0, 4000000, 0, -1000)
    with gdal.GetDriverByName("GTiff").Create(str(path), 2, 2, 1, gdal.GDT_Byte) as raster:
        raster.SetGeoTransform(grid)
        raster.SetProjection(crs(ALBERS).ExportToWkt())
        raster.GetRasterBand(1).WriteArray(np.array([[1, 2], [3, 0]], dtype=np.uint8))
    native = (100, 0.5, 0, 40, 0, -0.5)
    weights = diagnostic.spatial_weights(path, native, (20, 20))
    np.testing.assert_allclose(weights.sum(axis=1), [2e-5, 1e-5], rtol=1e-12)
    np.testing.assert_allclose(
        diagnostic.aggregate(np.ones((1, 20, 20)) * 7, weights), [[14e-5, 7e-5]]
    )
    yy, xx = np.indices((20, 20))
    field = 2 * xx + 3 * yy + 10
    transform = osr.CoordinateTransformation(crs(ALBERS), crs("EPSG:4326"))
    centres = [(500, 3999500), (1500, 3999500), (500, 3998500)]
    points = np.asarray(transform.TransformPoints(centres))
    cols = (points[:, 0] - native[0]) / native[1] - 0.5
    rows = (points[:, 1] - native[3]) / native[5] - 0.5
    expected = 2 * cols + 3 * rows + 10
    np.testing.assert_allclose(
        diagnostic.aggregate(field[None], weights),
        [[expected[:2].sum() * 1e-5, expected[2] * 1e-5]],
    )


def test_aggregation_rejects_missing_contributing_cells(diagnostic):
    weights = np.array([[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(
        diagnostic.aggregate(np.array([[[2.0, np.nan]]]), weights), [[2.0, 0.0]]
    )
    with pytest.raises(ValueError, match="Invalid contributing"):
        diagnostic.aggregate(np.array([[[np.nan, 2.0]]]), weights)
