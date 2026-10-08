"""Native versus aligned climate water-volume diagnostics on the same basin mask."""

import csv

import numpy as np
from osgeo import gdal, osr

from ecologyhydro.spatial import crs, windows


def native_weights(scope, native):
    gt = native.GetGeoTransform()
    width, height = native.RasterXSize, native.RasterYSize
    # Zero outside the basin must participate in average resampling.
    with gdal.Translate("", scope, format="MEM") as valid_scope:
        valid_scope.GetRasterBand(1).DeleteNoDataValue()
        for window in windows(valid_scope):
            values = (valid_scope.ReadAsArray(*window) > 0).astype(np.uint8)
            valid_scope.GetRasterBand(1).WriteArray(values, window[0], window[1])
        with gdal.Warp(
            "",
            valid_scope,
            format="MEM",
            dstSRS=native.GetProjection(),
            outputBounds=[gt[0], gt[3] + gt[5] * height, gt[0] + gt[1] * width, gt[3]],
            width=width,
            height=height,
            resampleAlg="average",
            dstNodata=0,
            outputType=gdal.GDT_Float32,
        ) as fraction:
            fractions = fraction.ReadAsArray()
    source_crs = native.GetSpatialRef()
    source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source_crs, crs(scope.GetProjection()))
    x, y = np.meshgrid(gt[0] + np.arange(width + 1) * gt[1], gt[3] + np.arange(height + 1) * gt[5])
    points = np.array(transform.TransformPoints(np.column_stack([x.ravel(), y.ravel()])))[..., :2]
    points = points.reshape(height + 1, width + 1, 2)
    corners = [points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]]
    # Translation before shoelace avoids cancellation at large projected coordinates.
    origin = corners[0]
    corners = [point - origin for point in corners]
    area = np.zeros((height, width))
    for first, second in zip(corners, corners[1:] + corners[:1], strict=True):
        area += first[..., 0] * second[..., 1] - second[..., 0] * first[..., 1]
    return fractions * np.abs(area) / 2


def compare_resampling(report, scope_path, output):
    records = []
    weights_cache = {}
    with gdal.Open(str(scope_path)) as scope:
        pixel_area = abs(scope.GetGeoTransform()[1] * scope.GetGeoTransform()[5])
        for name, path in report["native"].items():
            if not name.startswith(("prec_", "bcpr_", "pet_")):
                continue
            with gdal.Open(path) as native:
                key = (
                    native.GetProjection(),
                    native.GetGeoTransform(),
                    native.RasterXSize,
                    native.RasterYSize,
                )
                if key not in weights_cache:
                    weights_cache[key] = native_weights(scope, native)
                weights = weights_cache[key]
                values = native.ReadAsArray()
                valid = native.GetRasterBand(1).GetMaskBand().ReadAsArray() != 0
                valid &= np.isfinite(values) & (values >= 0)
                native_volume = float((values[valid] * weights[valid]).sum(dtype=np.float64) / 1000)
                native_area = float(weights[valid].sum())
            aligned_volume, aligned_area = 0.0, 0.0
            with gdal.Open(report["aligned"][name]) as aligned:
                for window in windows(scope):
                    values = aligned.ReadAsArray(*window)
                    valid = scope.ReadAsArray(*window) > 0
                    valid &= aligned.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) != 0
                    valid &= np.isfinite(values) & (values >= 0)
                    aligned_area += int(valid.sum()) * pixel_area
                    aligned_volume += float(values[valid].sum(dtype=np.float64)) * pixel_area / 1000
            records.append(
                {
                    "variable_year": name,
                    "native_volume_m3": native_volume,
                    "aligned_volume_m3": aligned_volume,
                    "relative_volume_change": aligned_volume / native_volume - 1,
                    "native_fractional_area_km2": native_area / 1e6,
                    "aligned_area_km2": aligned_area / 1e6,
                }
            )
    with (output / "resampling.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    return {
        "method": "native fractional mask and equal-area cell quadrilaterals vs aligned basin sum",
        "maximum_absolute_relative_volume_change": max(
            abs(r["relative_volume_change"]) for r in records
        ),
        "limitation": "fractional coverage and curved geographic cell edges are approximated",
    }
