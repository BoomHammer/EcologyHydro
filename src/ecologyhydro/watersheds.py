"""Mainstem-only station matching, reusable D8 routing and nested partitions."""

import csv
import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pygeoprocessing.routing as routing
from osgeo import gdal, ogr, osr

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.config import project_root
from ecologyhydro.spatial import OPTIONS, crs, windows

LOGGER = logging.getLogger(__name__)


def mainstem_chain(records):
    """Trace the dominant outlet upstream using upstream area at each junction."""
    groups = defaultdict(list)
    for identifier, row in records.items():
        if row["ENDORHEIC"] == 0:
            groups[row["MAIN_RIV"]].append(identifier)
    if not groups:
        raise ValueError("No exorheic river network")
    outlet = max(groups, key=lambda key: max(records[i]["UPLAND_SKM"] for i in groups[key]))
    if outlet not in records:
        raise ValueError(f"Main river outlet is missing: {outlet}")
    upstream = defaultdict(list)
    for identifier in groups[outlet]:
        upstream[records[identifier]["NEXT_DOWN"]].append(identifier)
    chain, current = [], outlet
    while current:
        if current in chain:
            raise ValueError("Cycle in river network")
        chain.append(current)
        candidates = upstream[current]
        if not candidates:
            break
        ranked = sorted(candidates, key=lambda i: records[i]["UPLAND_SKM"], reverse=True)
        if len(ranked) > 1 and records[ranked[0]]["UPLAND_SKM"] == records[ranked[1]]["UPLAND_SKM"]:
            raise ValueError(f"Ambiguous mainstem junction at {current}")
        current = ranked[0]
    return chain


def nearest_on_line(point, coordinates):
    points = np.asarray(coordinates, dtype=float)[:, :2]
    starts, vectors = points[:-1], np.diff(points, axis=0)
    lengths = (vectors * vectors).sum(axis=1)
    fractions = np.divide(
        ((point - starts) * vectors).sum(axis=1),
        lengths,
        out=np.zeros_like(lengths),
        where=lengths > 0,
    ).clip(0, 1)
    projected = starts + fractions[:, None] * vectors
    distances = ((projected - point) ** 2).sum(axis=1)
    index = distances.argmin()
    return math.sqrt(distances[index]), projected[index]


def make_layer(path, name, projection, geometry_type, fields):
    output = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = output.CreateLayer(name, crs(projection), geometry_type)
    for field, kind in fields:
        layer.CreateField(ogr.FieldDefn(field, kind))
    return output, layer


def append_feature(layer, geometry, values):
    feature = ogr.Feature(layer.GetLayerDefn())
    if layer.GetGeomType() == ogr.wkbMultiPolygon and geometry.GetGeometryType() == ogr.wkbPolygon:
        geometry = ogr.ForceToMultiPolygon(geometry.Clone())
    feature.SetGeometry(geometry)
    for key, value in values.items():
        feature.SetField(key, value)
    layer.CreateFeature(feature)


def match_stations(network, stations, projection, limit, output):
    records, geometries = {}, {}
    with ogr.Open(str(network)) as source:
        layer = source.GetLayer()
        source_crs = layer.GetSpatialRef()
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source_crs, crs(projection))
        for feature in layer:
            identifier = int(feature["HYRIV_ID"])
            records[identifier] = {
                key: feature[key] for key in ("NEXT_DOWN", "MAIN_RIV", "UPLAND_SKM", "ENDORHEIC")
            }
            geometry = feature.GetGeometryRef().Clone()
            geometry.Transform(transform)
            if geometry.GetGeometryType() != ogr.wkbLineString:
                raise ValueError("Expected LineString river reaches")
            geometries[identifier] = geometry
    chain = mainstem_chain(records)
    vector, layer = make_layer(
        output / "mainstem.gpkg",
        "mainstem",
        projection,
        ogr.wkbLineString,
        [("river_id", ogr.OFTInteger64)],
    )
    for identifier in chain:
        append_feature(layer, geometries[identifier], {"river_id": identifier})
    vector.Close()
    transform = osr.CoordinateTransformation(crs("EPSG:4490"), crs(projection))
    matches = []
    with stations.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 1):
            point = np.array(transform.TransformPoint(float(row["X"]), float(row["Y"]))[:2])
            candidates = [(nearest_on_line(point, geometries[i].GetPoints()), i) for i in chain]
            (distance, snapped), identifier = min(candidates, key=lambda value: value[0][0])
            matches.append(
                {
                    "ws_id": index,
                    "station": row["测站"],
                    "longitude": float(row["X"]),
                    "latitude": float(row["Y"]),
                    "original_x": float(point[0]),
                    "original_y": float(point[1]),
                    "x": float(snapped[0]),
                    "y": float(snapped[1]),
                    "river_id": identifier,
                    "distance_m": float(distance),
                    "chain_index": chain.index(identifier),
                    "reference_km2": float(row["控制面积(平方千米)"]),
                    "review_required": bool(distance > limit),
                }
            )
    if any(
        a["chain_index"] <= b["chain_index"] for a, b in zip(matches, matches[1:], strict=False)
    ):
        raise ValueError("Station order disagrees with mainstem topology")
    with (output / "stations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=matches[0])
        writer.writeheader()
        writer.writerows(matches)
    # Portable station/mainstem map with original-to-matched displacement lines.
    points = [p for i in chain for p in geometries[i].GetPoints()]
    west, east = min(p[0] for p in points), max(p[0] for p in points)
    south, north = min(p[1] for p in points), max(p[1] for p in points)
    scale = min(1100 / (east - west), 650 / (north - south))

    def xy(x, y):
        return f"{40 + (x - west) * scale:.2f},{40 + (north - y) * scale:.2f}"

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="800">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for identifier in chain:
        coords = " ".join(xy(*p[:2]) for p in geometries[identifier].GetPoints())
        svg.append(f'<polyline points="{coords}" fill="none" stroke="#2879b0"/>')
    for row in matches:
        x, y = xy(row["x"], row["y"]).split(",")
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#c33"/>')
        svg.append(
            f'<text x="{x}" y="{float(y) - 8}" font-size="14">{escape(row["station"])}</text>'
        )
        coords = xy(row["original_x"], row["original_y"]) + " " + xy(row["x"], row["y"])
        svg.append(f'<polyline points="{coords}" stroke="#c33"/>')
    svg.append("</svg>")
    (output / "stations.svg").write_text("\n".join(svg), encoding="utf-8")
    return {
        "method": "dominant exorheic outlet; maximum upstream-area chain; station order verified",
        "reaches": len(chain),
        "outlet_id": chain[0],
        "stations": matches,
        "review_required": any(r["review_required"] for r in matches),
        "coordinate_operation": "CGCS2000 to WGS84 Albers; metre-scale datum approximation",
    }


def routing_step(source, target, operation, output):
    work = output / "scratch"
    work.mkdir()
    try:
        if operation == "fill":
            routing.fill_pits((str(source), 1), str(target), working_dir=str(work))
        elif operation == "direction":
            routing.flow_dir_d8((str(source), 1), str(target), working_dir=str(work))
        else:
            routing.flow_accumulation_d8((str(source), 1), str(target))
    finally:
        shutil.rmtree(work)


def make_outlets(mainstem, stations, accumulation, radius, output):
    with gdal.Open(str(accumulation)) as raster:
        gt = raster.GetGeoTransform()
        inverse = gdal.InvGeoTransform(gt)
        vector, layer = make_layer(
            output / "outlets.gpkg",
            "outlets",
            raster.GetProjection(),
            ogr.wkbPoint,
            [("ws_id", ogr.OFTInteger), ("station", ogr.OFTString)],
        )
        records = []
        with ogr.Open(str(mainstem)) as lines:
            for station in stations:
                cx, cy = gdal.ApplyGeoTransform(inverse, station["x"], station["y"])
                cells = math.ceil(radius / gt[1])
                x0, y0 = max(0, int(cx) - cells), max(0, int(cy) - cells)
                width = min(raster.RasterXSize - x0, 2 * cells + 1)
                height = min(raster.RasterYSize - y0, 2 * cells + 1)
                mask = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
                origin = gdal.ApplyGeoTransform(gt, x0, y0)
                mask.SetGeoTransform((origin[0], gt[1], 0, origin[1], 0, gt[5]))
                mask.SetProjection(raster.GetProjection())
                river_layer = lines.GetLayer()
                river_layer.ResetReading()
                gdal.RasterizeLayer(
                    mask, [1], river_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"]
                )
                data = raster.ReadAsArray(x0, y0, width, height)
                yy, xx = np.indices(data.shape)
                squared = ((xx + x0 + 0.5 - cx) * gt[1]) ** 2 + ((yy + y0 + 0.5 - cy) * gt[5]) ** 2
                eligible = (mask.ReadAsArray() == 1) & (squared <= radius**2) & (data > 0)
                if not eligible.any():
                    raise ValueError(f"No mainstem-intersecting outlet for {station['station']}")
                y, x = np.unravel_index(np.where(eligible, data, -1).argmax(), data.shape)
                px, py = gdal.ApplyGeoTransform(gt, x0 + int(x) + 0.5, y0 + int(y) + 0.5)
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint_2D(px, py)
                append_feature(
                    layer, point, {"ws_id": station["ws_id"], "station": station["station"]}
                )
                records.append(
                    {
                        **station,
                        "outlet_x": px,
                        "outlet_y": py,
                        "pixel_x": x0 + int(x),
                        "pixel_y": y0 + int(y),
                        "outlet_shift_m": float(np.sqrt(squared[y, x])),
                        "accumulated_km2": float(data[y, x] * abs(gt[1] * gt[5]) / 1e6),
                    }
                )
        vector.Close()
    (output / "outlets.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"stations": records, "constraint": "only pixels intersecting identified mainstem"}


def delineate(direction, outlets, output):
    work = output / "scratch"
    try:
        routing.delineate_watersheds_d8(
            (str(direction), 1),
            str(outlets),
            str(output / "watersheds.gpkg"),
            working_dir=str(work),
        )
    finally:
        if work.exists():
            shutil.rmtree(work)


def partition(watersheds, template, stations, output):
    geometries = {}
    with ogr.Open(str(watersheds)) as dataset:
        for feature in dataset.GetLayer():
            geometries[int(feature["ws_id"])] = feature.GetGeometryRef().Clone()
    expected = {int(s["ws_id"]) for s in stations}
    if set(geometries) != expected:
        raise ValueError("Missing delineated stations")
    checks = []
    for row in stations:
        identifier = row["ws_id"]
        area = geometries[identifier].GetArea() / 1e6
        outside = (
            0
            if identifier == 1
            else geometries[identifier - 1].Difference(geometries[identifier]).GetArea() / 1e6
        )
        checks.append(
            {
                "station": row["station"],
                "ws_id": identifier,
                "area_km2": area,
                "reference_km2": row["reference_km2"],
                "difference_km2": area - row["reference_km2"],
                "upstream_outside_km2": outside,
            }
        )
    nested = all(row["upstream_outside_km2"] < 1e-6 for row in checks)
    (output / "area_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not nested:
        return {
            "accepted": False,
            "reason": "D8 catchments are not nested; no partition published",
            "areas": checks,
        }
    with gdal.Open(str(template)) as reference:
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(output / "zones.tif"),
            reference.RasterXSize,
            reference.RasterYSize,
            1,
            gdal.GDT_Byte,
            options=OPTIONS,
        )
        dataset.SetGeoTransform(reference.GetGeoTransform())
        dataset.SetProjection(reference.GetProjection())
        dataset.GetRasterBand(1).SetNoDataValue(0)
        memory = ogr.GetDriverByName("Memory").CreateDataSource("")
        layer = memory.CreateLayer("basins", reference.GetSpatialRef(), ogr.wkbMultiPolygon)
        layer.CreateField(ogr.FieldDefn("ws_id", ogr.OFTInteger))
        # Downstream first, upstream overwrites: each pixel belongs to one incremental zone.
        for identifier in sorted(geometries, reverse=True):
            append_feature(layer, geometries[identifier], {"ws_id": identifier})
        gdal.RasterizeLayer(dataset, [1], layer, options=["ATTRIBUTE=ws_id"])
        counts = np.zeros(len(stations) + 1, dtype=np.int64)
        for window in windows(dataset):
            counts += np.bincount(dataset.ReadAsArray(*window).ravel(), minlength=len(counts))
        dataset = None
        vector, layer = make_layer(
            output / "zones.gpkg",
            "zones",
            reference.GetProjection(),
            ogr.wkbMultiPolygon,
            [("ws_id", ogr.OFTInteger)],
        )
        for identifier in sorted(geometries):
            geometry = (
                geometries[identifier]
                if identifier == 1
                else geometries[identifier].Difference(geometries[identifier - 1])
            )
            append_feature(layer, geometry, {"ws_id": identifier})
        vector.Close()
    with (output / "zone_station.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["zone_id", "station_id", "station", "zone_pixels"])
        for station in stations:
            for zone in range(1, station["ws_id"] + 1):
                writer.writerow([zone, station["ws_id"], station["station"], counts[zone]])
    return {
        "accepted": True,
        "areas": checks,
        "zone_pixels": counts.tolist(),
        "limitation": "Reference areas diagnostic only; full upstream coverage needs edge checks",
    }


def prepare_watersheds(config, report):
    from ecologyhydro.conditioning import condition_dem, enforce_channel

    root = project_root()
    recipe = report["recipe"]
    cache = ArtifactCache(config.paths.cache / "m2")
    code = [Path(__file__), Path(__file__).with_name("cache.py")]
    network = root / recipe["river_network"]
    inputs = [network.with_suffix(suffix) for suffix in (".shp", ".shx", ".dbf", ".prj")]
    matched = cache.build(
        "routing/mainstem",
        [*inputs, root / recipe["stations"], *code],
        {"crs": report["grid"]["crs"], "limit": recipe["station_review_distance_m"]},
        lambda out: match_stations(
            network,
            root / recipe["stations"],
            report["grid"]["crs"],
            recipe["station_review_distance_m"],
            out,
        ),
    )
    match_report = json.loads((matched / "manifest.json").read_text(encoding="utf-8"))["details"]
    if match_report["review_required"]:
        raise ValueError(f"Station distance exceeds review limit; see {matched}")
    dem = Path(report["aligned"]["dem"])
    conditioned = cache.build(
        "routing/conditioned_dem",
        [
            dem,
            matched / "manifest.json",
            *inputs,
            Path(__file__).with_name("conditioning.py"),
            *code,
        ],
        {"burn_depth_m": 5},
        lambda out: condition_dem(dem, matched / "mainstem.gpkg", network, out),
    )
    previous = conditioned / "dem.tif"
    artifacts = {"mainstem": str(matched), "conditioning": str(conditioned)}
    for name in ("fill", "direction", "accumulation"):
        current = cache.build(
            f"routing/{name}",
            [previous, *code],
            {"method": name},
            lambda out, src=previous, op=name: routing_step(src, out / "data.tif", op, out),
        )
        previous = current / "data.tif"
        if name == "direction":
            enforced = cache.build(
                "routing/enforced_direction",
                [
                    previous,
                    conditioned / "manifest.json",
                    Path(__file__).with_name("conditioning.py"),
                    *code,
                ],
                {},
                lambda out, src=previous: enforce_channel(
                    src, conditioned / "channel_cells.npy", out
                ),
            )
            previous = enforced / "data.tif"
        artifacts[name] = str(previous)
    outlets = cache.build(
        "routing/outlets",
        [matched / "manifest.json", previous, *code],
        {"radius": recipe["outlet_search_radius_m"]},
        lambda out: make_outlets(
            matched / "mainstem.gpkg",
            match_report["stations"],
            previous,
            recipe["outlet_search_radius_m"],
            out,
        ),
    )
    basins = cache.build(
        "routing/watersheds",
        [Path(artifacts["direction"]), outlets / "manifest.json", *code],
        {},
        lambda out: delineate(Path(artifacts["direction"]), outlets / "outlets.gpkg", out),
    )
    zones = cache.build(
        "routing/partitions",
        [basins / "manifest.json", matched / "manifest.json", dem, *code],
        {},
        lambda out: partition(basins / "watersheds.gpkg", dem, match_report["stations"], out),
    )
    artifacts.update(outlets=str(outlets), watersheds=str(basins), partitions=str(zones))
    report["routing"] = artifacts
    report["cache_events"].extend(cache.events)
    (cache.root / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return artifacts
