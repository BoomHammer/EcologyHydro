"""Independent reference alarms; never fit catchment boundaries to tabular totals."""

import csv
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

from ecologyhydro.spatial import windows

REGIONS = [
    ("龙羊峡以上", 0, 2),
    ("龙羊峡至兰州", 2, 3),
    ("兰州至头道拐", 3, 6),
    ("头道拐至龙门", 6, 7),
    ("龙门至三门峡", 7, 8),
    ("三门峡至花园口", 8, 9),
    ("花园口以下", 9, 11),
]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def discrepancy(actual, reference, threshold):
    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(actual) or actual < 0:
        raise ValueError("Reference comparisons require positive references and valid totals")
    difference = actual / reference - 1
    return {"relative_difference": difference, "review_required": abs(difference) > threshold}


def precipitation_diagnostics(row, reference_area, thresholds):
    area = float(row["area_km2"])
    if area <= 0 or reference_area <= 0:
        raise ValueError("Precipitation comparison areas must be positive")
    volume = float(row["volume_1e8_m3"])
    reference_volume = float(row["observed_1e8_m3"])
    depth = volume * 1e5 / area
    reference_depth = reference_volume * 1e5 / reference_area
    area_check = discrepancy(area, reference_area, thresholds["area_relative_difference"])
    volume_check = discrepancy(
        volume, reference_volume, thresholds["precipitation_volume_relative_difference"]
    )
    depth_check = discrepancy(
        depth, reference_depth, thresholds["precipitation_depth_relative_difference"]
    )
    incomplete = float(row["valid_fraction"]) != 1
    reasons = [
        name
        for name, check in (
            ("area", area_check),
            ("precipitation_volume", volume_check),
            ("precipitation_depth", depth_check),
        )
        if check["review_required"]
    ]
    if incomplete:
        reasons.append("incomplete_climate_coverage")
    return {
        "region": row["region"],
        "year": int(row["year"]),
        "variable": row["variable"],
        "model_area_km2": area,
        "reference_area_km2": reference_area,
        "model_volume_1e8_m3": volume,
        "reference_volume_1e8_m3": reference_volume,
        "model_depth_mm": depth,
        "reference_depth_mm": reference_depth,
        "area_relative_difference": area_check["relative_difference"],
        "volume_relative_difference": volume_check["relative_difference"],
        "depth_relative_difference": depth_check["relative_difference"],
        "review_required": bool(reasons),
        "review_reasons": ";".join(reasons),
        "coverage_complete": not incomplete,
        "boundary_scope": "partial_downstream_extent"
        if row["region"] == "花园口以下"
        else "endpoint_proxy",
        "interpretation": "joint diagnostic; no boundary fitting or climate product selection",
    }


def reference_checks(zones_path, station_path, area_path, precipitation_path, thresholds, output):
    stations = read_rows(station_path)
    if len(stations) != 11:
        raise ValueError("Reference region mapping requires the documented eleven stations")
    reference = {
        r["水资源二级区"]: float(r["计算面积(万平方千米)"]) * 1e4 for r in read_rows(area_path)
    }
    counts = np.zeros(12, dtype=np.int64)
    with gdal.Open(str(zones_path)) as zones:
        for window in windows(zones):
            values = zones.ReadAsArray(*window)
            if np.any((values < 0) | (values > 11)):
                raise ValueError("Unexpected watershed IDs")
            counts += np.bincount(values.ravel().astype(int), minlength=12)
        pixel_area = abs(zones.GetGeoTransform()[1] * zones.GetGeoTransform()[5]) / 1e6
    cumulative = np.cumsum(counts[1:]) * pixel_area
    station_rows = []
    for i, row in enumerate(stations):
        expected = float(row["控制面积(平方千米)"])
        station_rows.append(
            {
                "station": row["测站"],
                "model_area_km2": float(cumulative[i]),
                "reference_area_km2": expected,
                **discrepancy(
                    float(cumulative[i]), expected, thresholds["area_relative_difference"]
                ),
            }
        )
    region_rows = []
    for name, start, end in REGIONS:
        area = float(counts[start + 1 : end + 1].sum() * pixel_area)
        region_rows.append(
            {
                "region": name,
                "model_area_km2": area,
                "reference_area_km2": reference[name],
                **discrepancy(area, reference[name], thresholds["area_relative_difference"]),
                "boundary_scope": "partial_downstream_extent" if end == 11 else "endpoint_proxy",
            }
        )
    rain = [
        precipitation_diagnostics(row, reference[row["region"]], thresholds)
        for row in read_rows(precipitation_path)
    ]
    for name, rows in (
        ("stations", station_rows),
        ("regions", region_rows),
        ("precipitation", rain),
    ):
        with (output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "thresholds": thresholds,
        "threshold_status": "engineering review triggers, not scientific acceptance tolerances",
        "station_alerts": [r["station"] for r in station_rows if r["review_required"]],
        "region_alerts": [r["region"] for r in region_rows if r["review_required"]],
        "precipitation_alert_rows": sum(r["review_required"] for r in rain),
        "review_required": any(r["review_required"] for r in [*station_rows, *region_rows, *rain]),
        "excluded_region": "黄河内流区 is never added to mainstem totals",
        "policy": "investigate large discrepancies jointly; never rescale geometry or rainfall",
        "validation_observations_used": False,
    }
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
