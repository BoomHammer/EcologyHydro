"""Delineate omitted sediment stations using supplied coordinates and existing D8 routing."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from osgeo import gdal, ogr, osr

from ecologyhydro.cache import fingerprint
from ecologyhydro.sediment import sediment_coverage
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import crs, windows
from ecologyhydro.watersheds import (
    append_feature,
    delineate,
    make_layer,
    make_outlets,
    nearest_on_line,
    partition,
)


def match_missing(observed, existing, mainstem, projection):
    """Match geographic points to the already audited dominant mainstem."""
    transform = osr.CoordinateTransformation(crs("EPSG:4490"), crs(projection))
    with ogr.Open(str(mainstem)) as vector:
        lines = [feature.GetGeometryRef().Clone() for feature in vector.GetLayer()]
    rows = []
    for _, row in observed.iterrows():
        if row["测站"] in existing:
            continue
        lon, lat = float(row.X), float(row.Y)
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Invalid station coordinates")
        point = np.array(transform.TransformPoint(lon, lat)[:2])
        distance, snapped = min(
            (nearest_on_line(point, line.GetPoints()) for line in lines), key=lambda item: item[0]
        )
        if distance > 10000:
            raise ValueError(f"Station too far from mainstem: {row['测站']}, {distance} m")
        rows.append(
            {
                "ws_id": len(rows) + 1,
                "station": row["测站"],
                "longitude": lon,
                "latitude": lat,
                "original_x": float(point[0]),
                "original_y": float(point[1]),
                "x": float(snapped[0]),
                "y": float(snapped[1]),
                "distance_m": float(distance),
                "reference_km2": float(row["控制面积(平方千米)"]),
            }
        )
    return rows


def merge_basins(original, additions, old_rows, new_rows, projection, output):
    """Preserve old boundaries and insert new basins in verified upstream order."""
    items = []
    for path, rows in ((original, old_rows), (additions, new_rows)):
        by_id = {int(row["ws_id"]): row for row in rows}
        with ogr.Open(str(path)) as dataset:
            for feature in dataset.GetLayer():
                row = by_id[int(feature["ws_id"])].copy()
                geometry = feature.GetGeometryRef().Clone()
                items.append((row, geometry))
    # Ordering is from supplied control areas; actual nesting must independently pass.
    items.sort(key=lambda item: item[0]["reference_km2"])
    vector, layer = make_layer(
        output / "watersheds.gpkg",
        "watersheds",
        projection,
        ogr.wkbMultiPolygon,
        [("ws_id", ogr.OFTInteger), ("station", ogr.OFTString)],
    )
    stations = []
    for identifier, (row, geometry) in enumerate(items, 1):
        row["previous_ws_id"] = row["ws_id"]
        row["ws_id"] = identifier
        stations.append(row)
        append_feature(layer, geometry, {"ws_id": identifier, "station": row["station"]})
    vector.Close()
    return stations


def summarize_baseline(output, sdr_run, grid_index):
    """Aggregate the unchanged official raster over the expanded station inventory."""
    coverage_config = output / "coverage_config.yaml"
    coverage_config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "grid_index": str(grid_index.resolve()),
                    "zones": str((output / "zones.tif").resolve()),
                }
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "comparison_config": str(coverage_config.resolve()),
        "zone_station": str((output / "zone_station.csv").resolve()),
    }
    coverage = sediment_coverage(sdr_run / "invest_sdr", Path.cwd(), settings)
    with (
        gdal.Open(str(output / "zones.tif")) as zones,
        gdal.Warp(
            "",
            str(sdr_run / "invest_sdr/usle.tif"),
            format="VRT",
            dstSRS=zones.GetProjection(),
            outputBounds=gdal.Info(zones, format="json")["cornerCoordinates"]["lowerLeft"]
            + gdal.Info(zones, format="json")["cornerCoordinates"]["upperRight"],
            width=zones.RasterXSize,
            height=zones.RasterYSize,
            dstNodata=float("nan"),
            resampleAlg="near",
        ) as usle,
    ):
        totals = np.zeros(256, dtype=np.float64)
        pixel_ha = abs(zones.GetGeoTransform()[1] * zones.GetGeoTransform()[5]) / 1e4
        for window in windows(zones):
            ids, values = zones.ReadAsArray(*window), usle.ReadAsArray(*window)
            valid = (ids > 0) & np.isfinite(values)
            totals += np.bincount(ids[valid], weights=values[valid], minlength=256) * pixel_ha
    members = pd.read_csv(output / "zone_station.csv")
    rows = []
    for identifier, group in members.groupby("station_id"):
        values = coverage[identifier]
        total = values["independent_raster_sum_t_yr"]
        rows.append(
            {
                "station": group.station.iloc[0],
                "ws_id": identifier,
                "hillslope_delivery_t_yr": total,
                "hillslope_delivery_100million_t_yr": total / 1e8,
                "usle_t_yr": float(totals[group.zone_id.to_numpy()].sum()),
                "status": "uncalibrated_hillslope_delivery_not_gauged_load",
                **values,
            }
        )
    result = pd.DataFrame(rows)
    original = pd.read_csv(sdr_run / "station_sediment.csv").set_index("station")
    aligned = result.set_index("station").loc[original.index]
    relative = abs(aligned.independent_raster_sum_t_yr / original.independent_raster_sum_t_yr - 1)
    if relative.max() > 1e-10:
        raise ValueError("Adding stations changed existing station totals")
    result.to_csv(output / "station_sediment.csv", index=False, encoding="utf-8-sig")
    return float(relative.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdr-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    gdal.UseExceptions()
    gdal.SetCacheMax(256 * 1024**2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    new = output / "added"
    new.mkdir()
    index_path = Path("project/repairs/soil_routing/m2_index.json")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    routing = index["routing"]
    old = Path("project/repairs/closed_routing_v3")
    old_rows = json.loads((old / "area_checks.json").read_text(encoding="utf-8"))
    observed_path = Path("data/Hydrology/输沙量2011-2024.csv")
    observed = pd.read_csv(observed_path)
    mainstem = Path(routing["mainstem"]) / "mainstem.gpkg"
    channel = Path(routing["conditioning"]) / "channel_cells.npy"
    matches = match_missing(
        observed, {r["station"] for r in old_rows}, mainstem, index["grid"]["crs"]
    )
    if not matches:
        raise ValueError("No missing sediment stations")
    outlets = make_outlets(
        mainstem, matches, old / "accumulation.tif", 2000, new, channel_path=channel
    )["stations"]
    print(json.dumps(outlets, ensure_ascii=False, indent=2), flush=True)
    delineate(old / "direction.tif", new / "outlets.gpkg", new)
    print("Additional catchments delineated", flush=True)
    stations = merge_basins(
        old / "watersheds.gpkg",
        new / "watersheds.gpkg",
        old_rows,
        outlets,
        index["grid"]["crs"],
        output,
    )
    result = partition(output / "watersheds.gpkg", old / "zones.tif", stations, output)
    if not result["accepted"]:
        raise ValueError("Expanded station catchments are not nested")
    areas = {r["station"]: r["area_km2"] for r in result["areas"]}
    for row in outlets:
        if abs(areas[row["station"]] - row["accumulated_km2"]) > 0.0625:
            raise ValueError("Delineated area and flow accumulation disagree")
        row["delineated_km2"] = areas[row["station"]]
        row["area_difference_percent"] = 100 * (row["delineated_km2"] / row["reference_km2"] - 1)
        row["original_to_outlet_m"] = float(
            np.hypot(row["outlet_x"] - row["original_x"], row["outlet_y"] - row["original_y"])
        )
    pd.DataFrame(outlets).to_csv(
        output / "added_station_checks.csv", index=False, encoding="utf-8-sig"
    )
    with gdal.Open(str(old / "zones.tif")) as a, gdal.Open(str(output / "zones.tif")) as b:
        for window in windows(a):
            if not np.array_equal(a.ReadAsArray(*window) > 0, b.ReadAsArray(*window) > 0):
                raise ValueError("Expanded inventory changed the overall reporting domain")
    unchanged = summarize_baseline(output, args.sdr_run, index_path)
    sources = [
        observed_path,
        index_path,
        old / "direction.tif",
        old / "accumulation.tif",
        old / "watersheds.gpkg",
        old / "area_checks.json",
        mainstem,
        channel,
        args.sdr_run / "invest_sdr/sed_export.tif",
        args.sdr_run / "invest_sdr/usle.tif",
        args.sdr_run / "invest_sdr/stream.tif",
        Path(__file__),
        Path("src/ecologyhydro/watersheds.py"),
        Path("src/ecologyhydro/sediment.py"),
    ]
    result.update(
        status="complete",
        added_stations=outlets,
        original_station_max_relative_change=unchanged,
        station_count=len(stations),
        observed_station_count=len(observed),
        elapsed_seconds=time.perf_counter() - started,
        policy="Supplied coordinates; nearest enforced mainstem; no area/sediment fitting",
        coordinate_assumption="Same as existing workflow: CGCS2000 geographic to WGS84 Albers",
        sources=[fingerprint(path) for path in sources],
        references=[
            "https://pygeoprocessing.readthedocs.io/en/latest/api/pygeoprocessing.routing.html"
        ],
    )
    write_json(output / "summary.json", result)
    print(f"Complete: {output}; {len(stations)} station catchments", flush=True)


if __name__ == "__main__":
    main()
