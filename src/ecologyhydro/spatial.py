"""Windowed raster preparation on an explicit equal-area grid."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

ALBERS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs"
OPTIONS = ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER", "NUM_THREADS=1"]
NODATA = -9999.0


def crs(value):
    result = osr.SpatialReference()
    result.SetFromUserInput(value)
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def bounds(path, target_crs):
    with gdal.Open(str(path)) as dataset:
        source = dataset.GetSpatialRef()
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source, crs(target_crs))
        gt = dataset.GetGeoTransform()
        points = []
        for t in np.linspace(0, 1, 101):
            for x, y in ((t, 0), (t, 1), (0, t), (1, t)):
                px, py = gdal.ApplyGeoTransform(
                    gt, x * dataset.RasterXSize, y * dataset.RasterYSize
                )
                points.append(transform.TransformPoint(px, py))
    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    ]


def make_grid(reference, resolution):
    extent = bounds(reference, ALBERS)
    extent = [math.floor(v / resolution) * resolution for v in extent[:2]] + [
        math.ceil(v / resolution) * resolution for v in extent[2:]
    ]
    return {
        "crs": ALBERS,
        "bounds": extent,
        "resolution": resolution,
        "width": round((extent[2] - extent[0]) / resolution),
        "height": round((extent[3] - extent[1]) / resolution),
    }


def warp(source, target, grid, method="bilinear", source_crs=None):
    kwargs = {"srcSRS": source_crs} if source_crs else {}
    virtual = gdal.Translate("", str(source), format="VRT")
    # Palette metadata is not meaningful for the signed, nodata-aware model input.
    if virtual.GetRasterBand(1).GetColorTable() is not None:
        virtual.GetRasterBand(1).SetColorTable(None)
    with gdal.Warp(
        str(target),
        virtual,
        format="GTiff",
        dstSRS=grid["crs"],
        outputBounds=grid["bounds"],
        width=grid["width"],
        height=grid["height"],
        resampleAlg=method,
        dstNodata=NODATA,
        outputType=gdal.GDT_Int32 if method == "near" else gdal.GDT_Float32,
        warpMemoryLimit=128,
        multithread=False,
        creationOptions=OPTIONS,
        **kwargs,
    ) as dataset:
        if dataset is None:
            raise RuntimeError(f"Warp failed: {source}")


def write_raster(path, array, transform, projection, nodata=NODATA):
    with gdal.GetDriverByName("GTiff").Create(
        str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Float32, options=OPTIONS
    ) as dataset:
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(crs(projection).ExportToWkt())
        dataset.GetRasterBand(1).SetNoDataValue(nodata)
        dataset.GetRasterBand(1).WriteArray(array)


def windows(dataset, rows=128):
    for y in range(0, dataset.RasterYSize, rows):
        yield (0, y, dataset.RasterXSize, min(rows, dataset.RasterYSize - y))


def statistics(path, categorical=False):
    count = missing = 0
    total = 0.0
    minimum, maximum = math.inf, -math.inf
    classes = Counter()
    with gdal.Open(str(path)) as dataset:
        band = dataset.GetRasterBand(1)
        for window in windows(dataset):
            values = band.ReadAsArray(*window)
            valid = (band.GetMaskBand().ReadAsArray(*window) != 0) & np.isfinite(values)
            data = values[valid]
            count += int(data.size)
            missing += int(values.size - data.size)
            if data.size:
                total += float(data.sum(dtype=np.float64))
                minimum = min(minimum, float(data.min()))
                maximum = max(maximum, float(data.max()))
                if categorical:
                    codes, counts = np.unique(data, return_counts=True)
                    classes.update(dict(zip(map(int, codes), map(int, counts), strict=True)))
        return {
            "valid": count,
            "missing": missing,
            "missing_fraction": missing / (count + missing),
            "min": minimum if count else None,
            "max": maximum if count else None,
            "mean": total / count if count else None,
            "classes": dict(classes),
            "size": [dataset.RasterXSize, dataset.RasterYSize],
            "transform": dataset.GetGeoTransform(),
            "projection": dataset.GetProjection(),
        }


def crop_soil(source: Path, target: Path, geographic_bounds):
    with gdal.Open(str(source)) as dataset:
        gt = dataset.GetGeoTransform()
        west, south, east, north = geographic_bounds
        x0 = max(0, math.floor((west - gt[0]) / gt[1]))
        x1 = min(dataset.RasterXSize, math.ceil((east - gt[0]) / gt[1]))
        y0 = max(0, math.floor((north - gt[3]) / gt[5]))
        y1 = min(dataset.RasterYSize, math.ceil((south - gt[3]) / gt[5]))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Soil and reference extent do not overlap")
        with gdal.Translate(
            str(target),
            dataset,
            srcWin=[x0, y0, x1 - x0, y1 - y0],
            outputSRS="EPSG:4326",
            creationOptions=OPTIONS,
        ):
            pass
        return {
            "source_pixels": dataset.RasterXSize * dataset.RasterYSize,
            "cropped_pixels": (x1 - x0) * (y1 - y0),
            "crs_assumption": "HWSD2 geographic WGS84; source has no CRS",
            "window": [x0, y0, x1 - x0, y1 - y0],
        }
