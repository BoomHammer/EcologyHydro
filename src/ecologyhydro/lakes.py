"""Shared vector lake overlay for paired land-cover experiments."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.config import project_root
from ecologyhydro.spatial import OPTIONS, crs, windows


def lake_mask(source, template, output):
    """Rasterize polygon interiors, preserving islands and using pixel centres."""
    with ogr.Open(str(source)) as vector, gdal.Open(str(template)) as reference:
        source_layer = vector.GetLayer()
        source_crs = source_layer.GetSpatialRef()
        if source_crs is None:
            raise ValueError("Lake polygons require an explicit CRS")
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_crs = crs(reference.GetProjection())
        options = osr.CoordinateTransformationOptions()
        options.SetBallparkAllowed(False)
        transform = osr.CreateCoordinateTransformation(source_crs, target_crs, options)
        if transform is None:
            raise ValueError("No supported lake coordinate transformation")
        memory = ogr.GetDriverByName("Memory").CreateDataSource("")
        layer = memory.CreateLayer("lakes", target_crs, ogr.wkbMultiPolygon)
        count = repaired = 0
        area = 0.0
        for feature in source_layer:
            original = feature.GetGeometryRef()
            if original is None or original.IsEmpty():
                raise ValueError(f"Empty lake polygon: {feature.GetFID()}")
            geometry = original.Clone()
            if not geometry.IsValid():
                geometry = geometry.MakeValid()
                repaired += 1
            if ogr.GT_Flatten(geometry.GetGeometryType()) not in (
                ogr.wkbPolygon,
                ogr.wkbMultiPolygon,
            ):
                raise ValueError("Lake geometry must remain polygonal after repair")
            if geometry.Transform(transform) != 0 or not geometry.IsValid():
                raise ValueError("Invalid transformed lake geometry")
            area += geometry.GetArea()
            copied = ogr.Feature(layer.GetLayerDefn())
            copied.SetGeometry(ogr.ForceToMultiPolygon(geometry))
            layer.CreateFeature(copied)
            count += 1
        if not count:
            raise ValueError("No lake polygons")
        with gdal.GetDriverByName("GTiff").Create(
            str(output / "mask.tif"),
            reference.RasterXSize,
            reference.RasterYSize,
            1,
            gdal.GDT_Byte,
            options=OPTIONS,
        ) as target:
            target.SetGeoTransform(reference.GetGeoTransform())
            target.SetProjection(reference.GetProjection())
            target.GetRasterBand(1).SetNoDataValue(0)
            target.GetRasterBand(1).Fill(0)
            gdal.RasterizeLayer(target, [1], layer, burn_values=[1])
            pixels = sum(int(np.count_nonzero(target.ReadAsArray(*w))) for w in windows(target))
        if not pixels:
            raise ValueError("Lake polygons do not intersect the model grid")
        pixel_area = abs(reference.GetGeoTransform()[1] * reference.GetGeoTransform()[5])
        return {
            "features": count,
            "repaired_features": repaired,
            "source_crs": source_crs.ExportToWkt(),
            "polygon_area_sum_km2": area / 1e6,
            "mask_pixels": pixels,
            "mask_area_km2": pixels * pixel_area / 1e6,
            "rasterization": "pixel centre; polygon holes preserved; overlaps counted once",
            "scope": "supplied polygons only; absence is not proof of land or drainage",
            "shoreline_year": "not established; fixed shared baseline mask",
        }


def overlay_lakes(source, mask_path, water_code, output):
    before = Counter()
    pixels = changed = filled = 0
    with gdal.Open(str(source)) as original, gdal.Open(str(mask_path)) as mask:
        if (
            original.GetGeoTransform() != mask.GetGeoTransform()
            or original.GetProjection() != mask.GetProjection()
            or (original.RasterXSize, original.RasterYSize) != (mask.RasterXSize, mask.RasterYSize)
        ):
            raise ValueError("Lake mask and land cover must share the same grid")
        with gdal.GetDriverByName("GTiff").CreateCopy(
            str(output / "data.tif"),
            original,
            options=OPTIONS,
        ) as target:
            for window in windows(original):
                inside = mask.ReadAsArray(*window) == 1
                values = original.ReadAsArray(*window)
                codes, counts = np.unique(values[inside], return_counts=True)
                before.update(dict(zip(map(int, codes), map(int, counts), strict=True)))
                pixels += int(inside.sum())
                changed += int(np.count_nonzero(inside & (values != water_code)))
                valid = original.GetRasterBand(1).GetMaskBand().ReadAsArray(*window) != 0
                filled += int(np.count_nonzero(inside & ~valid))
                values[inside] = water_code
                target.GetRasterBand(1).WriteArray(values, window[0], window[1])
        pixel_area = abs(original.GetGeoTransform()[1] * original.GetGeoTransform()[5])
    return {
        "water_code": water_code,
        "lake_pixels": pixels,
        "changed_pixels": changed,
        "changed_area_km2": changed * pixel_area / 1e6,
        "filled_nodata_pixels": filled,
        "classes_before": dict(before),
        "policy": "supplied lake polygons override classes; outside remains original",
    }


def prepare_lake_overlay(config, report):
    settings = report["recipe"].get("lake_overlay")
    if not settings:
        return
    source = project_root() / settings["vector"]
    originals = report.setdefault("original_aligned_landcover", {})
    for product in report["recipe"]["landcover"]:
        originals.setdefault(product, report["aligned"][f"lulc_{product}"])
    signatures = []
    for product, code in settings["water_codes"].items():
        with (Path(report["biophysical"]) / f"{product}.csv").open(
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            row = next((r for r in csv.DictReader(stream) if int(r["lucode"]) == code), None)
        if row is None or int(row["lulc_veg"]) != 0 or row["group"] != "water":
            raise ValueError(f"Overlay code is not an AWY water class: {product}/{code}")
        signatures.append(float(row["kc"]))
    if set(settings["water_codes"]) != set(originals) or len(set(signatures)) != 1:
        raise ValueError("Both products must share the lake mask and water evaporation coefficient")
    cache = ArtifactCache(config.paths.cache / "m2")
    code = [
        Path(__file__),
        Path(__file__).with_name("spatial.py"),
        Path(__file__).with_name("cache.py"),
    ]
    mask = cache.build(
        "landcover/lake_mask",
        [source.with_suffix(s) for s in (".shp", ".shx", ".dbf", ".prj")]
        + [Path(next(iter(originals.values()))), *code],
        {"grid": report["grid"], "rule": "pixel_centre"},
        lambda out: lake_mask(source, next(iter(originals.values())), out),
    )
    overlays = {}
    for product, water_code in settings["water_codes"].items():
        artifact = cache.build(
            f"landcover/lake_overlay/{product}",
            [Path(originals[product]), mask / "manifest.json", *code],
            {"water_code": water_code},
            lambda out, p=product, c=water_code: overlay_lakes(
                originals[p], mask / "mask.tif", c, out
            ),
        )
        report["aligned"][f"lulc_{product}"] = str(artifact / "data.tif")
        overlays[product] = {
            "artifact": str(artifact),
            **json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))["details"],
        }
    report["lake_overlay"] = {"mask": str(mask), "products": overlays}
    report["cache_events"].extend(cache.events)
