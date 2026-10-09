"""Cross-check modeled catchments against the supplied river network topology."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

from ecologyhydro.aggregation import write_csv
from ecologyhydro.config import project_root
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import crs
from ecologyhydro.water_balance import read_csv


def main():
    gdal.UseExceptions()
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=root / "project/cache/m2/latest.json")
    parser.add_argument("--output", type=Path, default=root / "project/diagnostics/deep_review")
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    matched = read_csv(Path(index["routing"]["mainstem"]) / "stations.csv")
    station_reaches = {int(r["river_id"]): int(r["ws_id"]) for r in matched}
    with (
        gdal.Open(str(Path(index["routing"]["partitions"]) / "zones.tif")) as ds,
        ogr.Open(str(root / index["recipe"]["river_network"])) as source,
    ):
        zones = ds.ReadAsArray()
        inverse = gdal.InvGeoTransform(ds.GetGeoTransform())
        transform = osr.CoordinateTransformation(crs("EPSG:4326"), crs(ds.GetProjection()))

        def zone_at(lon, lat):
            x, y, _ = transform.TransformPoint(lon, lat)
            col, row = np.floor(gdal.ApplyGeoTransform(inverse, x, y)).astype(int)
            return (
                int(zones[row, col])
                if 0 <= row < zones.shape[0] and 0 <= col < zones.shape[1]
                else -1
            )

        records = {}
        for feature in source.GetLayer():
            geometry = feature.GetGeometryRef()
            # Interior vertices reduce endpoint-at-divide ambiguity.
            point = geometry.GetPoint(geometry.GetPointCount() // 2)
            identifier = int(feature["HYRIV_ID"])
            records[identifier] = {
                "river_id": identifier,
                "next_down": int(feature["NEXT_DOWN"]),
                "endorheic": int(feature["ENDORHEIC"]),
                "upland_km2": float(feature["UPLAND_SKM"]),
                "longitude": point[0],
                "latitude": point[1],
                "model_zone": zone_at(*point[:2]),
            }
        network_zone = {}
        for identifier in records:
            current, seen = identifier, set()
            while (
                current in records
                and current not in network_zone
                and current not in station_reaches
            ):
                if current in seen:
                    raise ValueError("Cycle in supplied HydroRIVERS")
                seen.add(current)
                current = records[current]["next_down"]
            expected = station_reaches.get(current, network_zone.get(current, 0))
            for item in seen:
                network_zone[item] = expected
            network_zone[identifier] = expected
        suspect = []
        for identifier, row in records.items():
            expected = network_zone[identifier]
            if (row["model_zone"] > 0 or expected > 0) and row["model_zone"] != expected:
                suspect.append({**row, "network_first_station": expected})
        locations = [
            {"label": "Qinghai_Lake_interior_diagnostic", "lon": 100.15, "lat": 36.85},
            {"label": "Qinghai_Lake_west_diagnostic", "lon": 99.85, "lat": 36.85},
        ]
        for item in locations:
            item["model_zone"] = zone_at(item["lon"], item["lat"])
    if suspect:
        write_csv(output / "river_topology_disagreements.csv", suspect)
    report = {
        "reaches": len(records),
        "disagreement_sample_points": len(suspect),
        "mapped_endorheic_reaches": sum(r["endorheic"] == 1 for r in records.values()),
        "endorheic_reaches_in_model": sum(
            r["endorheic"] == 1 and r["model_zone"] > 0 for r in records.values()
        ),
        "mapped_exorheic_reaches_excluded": sum(
            r["model_zone"] <= 0 and r["network_first_station"] > 0 for r in suspect
        ),
        "disagreement_counts_by_model_zone": dict(Counter(r["model_zone"] for r in suspect)),
        "largest_disagreements": sorted(suspect, key=lambda r: r["upland_km2"], reverse=True)[:20],
        "lake_point_checks": locations,
        "major_tributary_point_checks": [
            {**records[i], "network_first_station": network_zone[i]}
            for i in (40494176, 40512176, 40473554)
            if i in records
        ],
        "limitation": "Point/network consistency diagnostic, not an independent catchment polygon. "
        "Missing downstream links in clipped network can produce expected zone zero.",
    }
    write_json(output / "routing_topology.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
