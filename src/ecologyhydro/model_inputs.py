"""Prepare AWY inputs without changing the shared M2 products."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

from ecologyhydro.cache import ArtifactCache, fingerprint
from ecologyhydro.spatial import NODATA, OPTIONS, windows
from ecologyhydro.watersheds import append_feature, make_layer


def verified_artifact(path):
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    item = next((row for row in manifest["outputs"] if Path(row["name"]) == Path(path.name)), None)
    if item is None or fingerprint(path)["sha256"] != item["sha256"]:
        raise ValueError(f"M2 output changed or is untracked: {path}")


def crop_scope(zones_path, vector_path, station_limit, output):
    with ogr.Open(str(vector_path)) as source:
        source_layer = source.GetLayer()
        vector, layer = make_layer(
            output / "watersheds.gpkg",
            "watersheds",
            source_layer.GetSpatialRef().ExportToWkt(),
            ogr.wkbMultiPolygon,
            [("ws_id", ogr.OFTInteger)],
        )
        ids = []
        for feature in source_layer:
            identifier = int(feature["ws_id"])
            if identifier <= station_limit:
                append_feature(layer, feature.GetGeometryRef(), {"ws_id": identifier})
                ids.append(identifier)
        if sorted(ids) != list(range(1, station_limit + 1)):
            raise ValueError("Incomplete or duplicate scope zone IDs")
        west, east, south, north = layer.GetExtent()
        vector.Close()
    with gdal.Open(str(zones_path)) as source:
        gt = source.GetGeoTransform()
        x0 = math.floor((west - gt[0]) / gt[1] + 1e-7)
        y0 = math.floor((north - gt[3]) / gt[5] + 1e-7)
        x1 = math.ceil((east - gt[0]) / gt[1] - 1e-7)
        y1 = math.ceil((south - gt[3]) / gt[5] - 1e-7)
        window = [x0, y0, x1 - x0, y1 - y0]
        if x0 < 0 or y0 < 0 or x1 > source.RasterXSize or y1 > source.RasterYSize:
            raise ValueError("Watersheds exceed zone raster")
        with gdal.Translate(
            str(output / "zones.tif"), source, srcWin=window, creationOptions=OPTIONS
        ) as target:
            for block in windows(target):
                values = target.ReadAsArray(*block)
                values[values > station_limit] = 0
                target.GetRasterBand(1).WriteArray(values, block[0], block[1])
    return {"source_window": window, "station_limit": station_limit}


def mask_input(source_path, zones_path, source_window, output, valid_classes=None):
    with (
        gdal.Open(str(zones_path)) as zones,
        gdal.Open(str(source_path)) as source,
        gdal.Translate(
            str(output), source, srcWin=source_window, noData=NODATA, creationOptions=OPTIONS
        ) as target,
    ):
        if (
            target.GetGeoTransform() != zones.GetGeoTransform()
            or target.GetProjection() != zones.GetProjection()
            or (target.RasterXSize, target.RasterYSize) != (zones.RasterXSize, zones.RasterYSize)
        ):
            raise ValueError("M2 inputs are not on the same grid")
        count = 0
        for block in windows(zones):
            inside = zones.ReadAsArray(*block) > 0
            values = target.ReadAsArray(*block)
            valid = np.isfinite(values) & (values != NODATA)
            if valid_classes is not None:
                valid &= np.isin(values, valid_classes)
            if np.any(inside & ~valid):
                raise ValueError(f"Invalid or unmapped model input within watershed: {source_path}")
            values[~inside] = NODATA
            target.GetRasterBand(1).WriteArray(values, block[0], block[1])
            count += int(inside.sum())
    return {"valid_pixels": count, "mask": "selected complete upstream zones; outside is NoData"}


def prepare_inputs(index_path, cache_root, year, landcover, scope):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if not index.get("readiness", {}).get("engineering_ready"):
        raise ValueError("M2 engineering acceptance is required")
    if landcover not in index["recipe"]["landcover"]:
        raise ValueError(f"Unknown landcover: {landcover}")
    selected = json.loads(
        (Path(index["quality"]["precipitation"]) / "manifest.json").read_text(encoding="utf-8")
    )["details"]["trial_selection"]
    limit = 1 if scope == "tangnaihai" else 11
    if scope not in {"tangnaihai", "full"}:
        raise ValueError("Unknown model scope")
    partition = Path(index["routing"]["partitions"])
    table = Path(index["biophysical"]) / f"{landcover}.csv"
    mapping = {
        "lulc_path": index["aligned"][f"lulc_{landcover}"],
        "precipitation_path": index["aligned"][f"{selected}_{year}"],
        "eto_path": index["aligned"][f"pet_{year}"],
        "pawc_path": index["aligned"]["pawc"],
        "depth_to_root_rest_layer_path": index["aligned"]["root_depth"],
    }
    for path in [
        *mapping.values(),
        partition / "zones.tif",
        partition / "zones.gpkg",
        table,
        partition / "zone_station.csv",
    ]:
        verified_artifact(path)
    cache = ArtifactCache(Path(cache_root) / "m3")
    code = [Path(__file__), Path(__file__).with_name("cache.py")]
    region = cache.build(
        "scopes",
        [partition / "zones.tif", partition / "zones.gpkg", *code],
        {"station_limit": limit},
        lambda out: crop_scope(partition / "zones.tif", partition / "zones.gpkg", limit, out),
    )
    window = json.loads((region / "manifest.json").read_text(encoding="utf-8"))["details"][
        "source_window"
    ]
    with table.open(encoding="utf-8-sig", newline="") as stream:
        classes = [int(row["lucode"]) for row in csv.DictReader(stream)]
    paths = {}
    for key, value in mapping.items():
        inputs = [Path(value), region / "manifest.json", *code]
        if key == "lulc_path":
            inputs.append(table)
        valid_classes = classes if key == "lulc_path" else None
        artifact = cache.build(
            f"masked/{key}",
            inputs,
            {},
            lambda out, src=value, codes=valid_classes: mask_input(
                src, region / "zones.tif", window, out / "data.tif", codes
            ),
        )
        paths[key] = str(artifact / "data.tif")
    paths.update(watersheds_path=str(region / "watersheds.gpkg"), biophysical_table_path=str(table))
    return {
        "args": paths,
        "zones": str(region / "zones.tif"),
        "zone_station": str(partition / "zone_station.csv"),
        "station_limit": limit,
        "precipitation": selected,
        "scope": scope,
        "year": year,
        "landcover": landcover,
        "cache_events": cache.events,
        "m2_readiness": index["readiness"],
    }
