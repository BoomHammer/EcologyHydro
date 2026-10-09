"""Preserve independently documented closed lakes as terminal drainage surfaces."""

import json

import numpy as np
from osgeo import gdal, ogr, osr
from scipy.ndimage import label

from ecologyhydro.spatial import OPTIONS, crs, windows


def inland_channel_mask(network, template, output):
    """Drain independently mapped inland channels outside the modeled exorheic domain."""
    with gdal.Open(str(template)) as reference, ogr.Open(str(network)) as vector:
        original = vector.GetLayer()
        original.SetAttributeFilter("ENDORHEIC = 1")
        source_crs = original.GetSpatialRef()
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        projected = crs(reference.GetProjection())
        transform = osr.CoordinateTransformation(source_crs, projected)
        memory = ogr.GetDriverByName("Memory").CreateDataSource("")
        layer = memory.CreateLayer("inland", projected, ogr.wkbLineString)
        reaches = 0
        for feature in original:
            geometry = feature.GetGeometryRef().Clone()
            geometry.Transform(transform)
            item = ogr.Feature(layer.GetLayerDefn())
            item.SetGeometry(geometry)
            layer.CreateFeature(item)
            reaches += 1
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
    return {
        "mapped_endorheic_reaches": reaches,
        "channel_pixels": pixels,
        "policy": "mapped inland drainage lines collect flow outside exorheic model domain",
        "limitation": "internal routing within excluded inland systems is not modeled",
        "source": "https://data.hydrosheds.org/file/technical-documentation/HydroRIVERS_TechDoc_v10.pdf",
    }


def closed_lake_mask(landcover, lakes, output):
    records = []
    with (
        gdal.Open(str(landcover)) as source,
        gdal.GetDriverByName("GTiff").Create(
            str(output / "mask.tif"),
            source.RasterXSize,
            source.RasterYSize,
            1,
            gdal.GDT_Byte,
            options=OPTIONS,
        ) as target,
    ):
        gt = source.GetGeoTransform()
        target.SetGeoTransform(gt)
        target.SetProjection(source.GetProjection())
        target.GetRasterBand(1).SetNoDataValue(0)
        target.GetRasterBand(1).Fill(0)
        inverse = gdal.InvGeoTransform(gt)
        transform = osr.CoordinateTransformation(crs("EPSG:4326"), crs(source.GetProjection()))
        for lake in lakes:
            if not lake.get("evidence"):
                raise ValueError("Closed lake status requires independent evidence")
            west, south, east, north = lake["search_bounds_lonlat"]
            corners = [
                gdal.ApplyGeoTransform(inverse, *transform.TransformPoint(x, y)[:2])
                for x in np.linspace(west, east, 21)
                for y in (south, north)
            ]
            corners += [
                gdal.ApplyGeoTransform(inverse, *transform.TransformPoint(x, y)[:2])
                for y in np.linspace(south, north, 21)
                for x in (west, east)
            ]
            x0, y0 = np.floor(np.min(corners, axis=0)).astype(int)
            x1, y1 = np.ceil(np.max(corners, axis=0)).astype(int)
            if min(x0, y0) < 0 or x1 > source.RasterXSize or y1 > source.RasterYSize:
                raise ValueError("Closed lake search window exceeds data coverage")
            window = tuple(map(int, (x0, y0, x1 - x0, y1 - y0)))
            components, _ = label(source.ReadAsArray(*window) == 80, structure=np.ones((3, 3)))
            px, py = np.floor(
                gdal.ApplyGeoTransform(inverse, *transform.TransformPoint(*lake["seed_lonlat"])[:2])
            ).astype(int)
            component = int(components[py - y0, px - x0])
            if not component:
                raise ValueError("Documented closed lake seed is not permanent water")
            mask = components == component
            if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
                raise ValueError("Closed water component touches search boundary")
            previous = target.ReadAsArray(*window)
            target.GetRasterBand(1).WriteArray(
                np.maximum(previous, mask).astype(np.uint8), int(x0), int(y0)
            )
            records.append(
                {
                    **lake,
                    "pixels": int(mask.sum()),
                    "lake_area_km2": float(mask.sum() * abs(gt[1] * gt[5]) / 1e6),
                }
            )
    result = {
        "lakes": records,
        "method": "documented endorheic status; connected Copernicus water geometry",
        "routing_policy": "terminal lake surface before filling; catchment determined by DEM",
        "not_used": "reference catchment areas and observed runoff",
    }
    (output / "lakes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def check_exclusion(zones_path, mask_path):
    count = included = 0
    with gdal.Open(str(zones_path)) as zones, gdal.Open(str(mask_path)) as mask:
        if (
            zones.GetGeoTransform() != mask.GetGeoTransform()
            or zones.GetProjection() != mask.GetProjection()
        ):
            raise ValueError("Closed lake check grids differ")
        for window in windows(zones):
            inside = mask.ReadAsArray(*window) == 1
            count += int(inside.sum())
            included += int(np.count_nonzero(inside & (zones.ReadAsArray(*window) > 0)))
    if not count or included:
        raise ValueError(
            f"Closed lake exclusion failed: {included} of {count} lake pixels in mainstem basins"
        )
    return {"closed_lake_pixels": count, "included_in_mainstem": included, "passed": True}
