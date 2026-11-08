"""Compact annual SDR products for fixed terrain, land cover and connectivity."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from osgeo import gdal, osr


def raster_grid(dataset):
    """Return exact north-up grid arguments accepted by GDAL Warp."""
    x, dx, rx, y, ry, dy = dataset.GetGeoTransform()
    if rx or ry or dx <= 0 or dy >= 0:
        raise ValueError("Expected a north-up raster")
    return {
        "dstSRS": dataset.GetProjection(),
        "outputBounds": [x, y + dy * dataset.RasterYSize, x + dx * dataset.RasterXSize, y],
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
    }


def native_factor(path, longitude, latitude, values):
    """Write a small geographic factor map, preserving missing values as NaN."""
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    # Float32 coordinate quantization biases median adjacent spacings over long axes.
    dx = (longitude[-1] - longitude[0]) / (len(longitude) - 1)
    dy = (latitude[-1] - latitude[0]) / (len(latitude) - 1)
    if dx <= 0 or dy == 0 or values.shape != (len(latitude), len(longitude)):
        raise ValueError("Invalid climate coordinates")
    if not np.allclose(np.diff(longitude), dx, atol=1e-4) or not np.allclose(
        np.diff(latitude), dy, atol=1e-4
    ):
        raise ValueError("Climate coordinates must be regularly spaced")
    spatial = osr.SpatialReference()
    spatial.ImportFromEPSG(4326)
    with gdal.GetDriverByName("GTiff").Create(
        str(path),
        len(longitude),
        len(latitude),
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=3"],
    ) as target:
        target.SetProjection(spatial.ExportToWkt())
        target.SetGeoTransform(
            (
                float(longitude[0] - dx / 2),
                float(dx),
                0,
                float(latitude.max() + abs(dy) / 2),
                0,
                -float(abs(dy)),
            )
        )
        band = target.GetRasterBand(1)
        band.SetNoDataValue(float("nan"))
        band.WriteArray(values[::-1] if dy > 0 else values)


def nan_warp(source, destination, grid, resampling="near"):
    """Convert source nodata to actual NaN, lazily, including on GDAL 3.10."""
    with gdal.Warp(
        str(destination),
        str(source),
        format="VRT",
        **grid,
        resampleAlg=resampling,
        dstNodata=float("nan"),
        outputType=gdal.GDT_Float32,
        errorThreshold=0,
        warpOptions=["NUM_THREADS=1"],
    ):
        pass


def multiply_vrt(sources, destination, template, unit):
    """Use built-in C multiplication; all source missing pixels must be NaN."""
    width, height = template.RasterXSize, template.RasterYSize
    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    ET.SubElement(root, "SRS").text = template.GetProjection()
    ET.SubElement(root, "GeoTransform").text = ",".join(map(str, template.GetGeoTransform()))
    band = ET.SubElement(
        root, "VRTRasterBand", dataType="Float32", band="1", subClass="VRTDerivedRasterBand"
    )
    ET.SubElement(band, "NoDataValue").text = "nan"
    ET.SubElement(band, "UnitType").text = unit
    ET.SubElement(band, "PixelFunctionType").text = "mul"
    for path in sources:
        source = ET.SubElement(band, "SimpleSource")
        ET.SubElement(source, "SourceFilename", relativeToVRT="0").text = str(Path(path).resolve())
        ET.SubElement(source, "SourceBand").text = "1"
        for name in ("SrcRect", "DstRect"):
            ET.SubElement(source, name, xOff="0", yOff="0", xSize=str(width), ySize=str(height))
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def summarize_year(year, directory):
    """Aggregate by disjoint zone; caller combines zones into nested catchments."""
    directory = Path(directory)
    sums = np.zeros((3, 256), dtype=np.float64)
    counts = np.zeros((2, 256), dtype=np.int64)
    with (
        gdal.Open(str(directory / "zones.vrt")) as zones,
        gdal.Open(str(directory / f"factor_{year}.vrt")) as factor,
        gdal.Open(str(directory / "base_export_nan.vrt")) as export,
        gdal.Open(str(directory / "base_usle_nan.vrt")) as usle,
        gdal.Open(str(directory / "base_r_nan.vrt")) as erosivity,
    ):
        pixel_ha = (
            abs(np.linalg.det(np.array(zones.GetGeoTransform())[[1, 2, 4, 5]].reshape(2, 2))) / 1e4
        )
        for y in range(0, zones.RasterYSize, 128):
            window = (0, y, zones.RasterXSize, min(128, zones.RasterYSize - y))
            ids = zones.ReadAsArray(*window).astype(np.int32)
            inside = ids > 0
            ratio = factor.ReadAsArray(*window)
            if np.any(inside & (~np.isfinite(ratio) | (ratio < 0))):
                raise ValueError(f"Missing/invalid CMFD factor inside reporting domain: {year}")
            for index, dataset in enumerate((export, usle, erosivity)):
                values = dataset.ReadAsArray(*window)
                valid = inside & np.isfinite(values)
                if np.any(values[valid] < 0):
                    raise ValueError("Negative baseline input")
                # Match Float32 virtual products; accumulate station totals in Float64.
                annual = values[valid] * ratio[valid]
                sums[index] += np.bincount(ids[valid], weights=annual, minlength=256)
                if index == 0:
                    counts[0] += np.bincount(ids[valid], minlength=256)
                if index == 2:
                    counts[1] += np.bincount(ids[valid], minlength=256)
    sums[:2] *= pixel_ha
    return year, sums, counts
