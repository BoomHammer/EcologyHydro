"""Windowed AWY volume accounting with explicit missing-area checks."""

import calendar
import csv
from contextlib import ExitStack

import numpy as np
from osgeo import gdal, ogr

from ecologyhydro.spatial import windows

# Float32 Fu evaluation can yield tiny negatives at the zero-yield limit.
# Keep real negative yields invalid and report every numerical correction.
YIELD_ROUNDOFF_TOLERANCE_MM = 0.0005


def read_observations(path, year):
    result = {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["测站"]
            if name in result:
                raise ValueError("Duplicate observation station")
            raw = row[str(year)].strip()
            value = float(raw) if raw else None
            status = "missing" if value is None else "valid"
            if value is not None and (not np.isfinite(value) or value <= 0):
                status = "invalid_observation"
            result[name] = {
                "raw": raw,
                "value": value if status == "valid" else None,
                "status": status,
            }
    return result


def zone_volumes(yield_path, zones_path, precipitation_path=None, aet_path=None):
    counts = np.zeros(256, dtype=np.int64)
    valid_counts = counts.copy()
    sums = np.zeros(256, dtype=np.float64)
    balance_error = 0.0
    roundoff_cells, roundoff_mm_pixels, minimum_raw_yield = 0, 0.0, 0.0
    with ExitStack() as stack:
        zones = stack.enter_context(gdal.Open(str(zones_path)))
        wyield = stack.enter_context(gdal.Open(str(yield_path)))
        related = {"yield": wyield}
        if precipitation_path is not None:
            related["precipitation"] = stack.enter_context(gdal.Open(str(precipitation_path)))
        if aet_path is not None:
            related["aet"] = stack.enter_context(gdal.Open(str(aet_path)))
        for name, raster in related.items():
            if (
                raster.GetGeoTransform() != zones.GetGeoTransform()
                or raster.GetProjection() != zones.GetProjection()
                or (raster.RasterXSize, raster.RasterYSize)
                != (zones.RasterXSize, zones.RasterYSize)
            ):
                raise ValueError(f"Mismatched aggregation grid: {name}")
        gt = zones.GetGeoTransform()
        pixel_area = abs(gt[1] * gt[5] - gt[2] * gt[4])
        for block in windows(zones):
            ids = zones.ReadAsArray(*block)
            inside = ids > 0
            values = wyield.ReadAsArray(*block)
            valid = inside & (wyield.GetRasterBand(1).GetMaskBand().ReadAsArray(*block) != 0)
            valid &= np.isfinite(values) & (values >= -YIELD_ROUNDOFF_TOLERANCE_MM)
            if np.any(inside & ~valid):
                raise ValueError(
                    "Missing/negative model yield inside watershed; no silent area loss"
                )
            counts += np.bincount(ids.ravel(), minlength=256)
            valid_counts += np.bincount(ids[valid], minlength=256)
            tiny_negative = valid & (values < 0)
            if tiny_negative.any():
                roundoff_cells += int(tiny_negative.sum())
                roundoff_mm_pixels -= float(values[tiny_negative].sum(dtype=float))
                minimum_raw_yield = min(minimum_raw_yield, float(values[tiny_negative].min()))
            sums += np.bincount(ids[valid], weights=np.maximum(values[valid], 0), minlength=256)
            if "precipitation" in related and "aet" in related:
                rain = related["precipitation"].ReadAsArray(*block)
                aet = related["aet"].ReadAsArray(*block)
                if inside.any():
                    error = float(np.max(np.abs((values + aet - rain)[inside])))
                    balance_error = max(balance_error, error)
                    if (
                        not np.isfinite(error)
                        or error > 0.002
                        or np.any(values[inside] > rain[inside] + 0.002)
                    ):
                        raise ValueError("AWY water-balance check failed")
    records = [
        {
            "zone_id": i,
            "pixels": int(counts[i]),
            "valid_pixels": int(valid_counts[i]),
            "area_km2": counts[i] * pixel_area / 1e6,
            "valid_area_km2": valid_counts[i] * pixel_area / 1e6,
            "volume_m3": sums[i] * pixel_area / 1000,
        }
        for i in range(1, 256)
        if counts[i]
    ]
    return records, {
        "maximum_water_balance_error_mm": balance_error,
        "pixel_area_m2": pixel_area,
        "yield_roundoff_tolerance_mm": YIELD_ROUNDOFF_TOLERANCE_MM,
        "negative_roundoff_cells": roundoff_cells,
        "negative_roundoff_correction_m3": roundoff_mm_pixels * pixel_area / 1000,
        "minimum_raw_yield_mm": minimum_raw_yield,
    }


def official_volume_check(records, vector_path):
    ours = {row["zone_id"]: row["volume_m3"] for row in records}
    comparisons = []
    with ogr.Open(str(vector_path)) as vector:
        for feature in vector.GetLayer():
            identifier, volume = int(feature["ws_id"]), feature["wyield_vol"]
            if volume is None or identifier not in ours:
                raise ValueError("Missing official watershed result")
            difference = abs(volume - ours[identifier])
            if not np.isclose(volume, ours[identifier], rtol=1e-5, atol=1):
                raise ValueError(f"Official and pixel-wise zone volumes differ: {identifier}")
            comparisons.append(
                {
                    "zone_id": identifier,
                    "official_m3": volume,
                    "pixel_m3": ours[identifier],
                    "absolute_difference_m3": difference,
                }
            )
    if len(comparisons) != len(records):
        raise ValueError("Official result zone count mismatch")
    return comparisons


def station_totals(
    records, mapping_path, observations, run_id, year, landcover, precipitation, limit
):
    zones = {row["zone_id"]: row for row in records}
    relationships = {}
    with open(mapping_path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            station_id, zone_id = int(row["station_id"]), int(row["zone_id"])
            if station_id > limit:
                continue
            entry = relationships.setdefault(station_id, {"name": row["station"], "zones": set()})
            if zone_id in entry["zones"]:
                raise ValueError("Duplicate zone-station relationship would double count volume")
            entry["zones"].add(zone_id)
    result = []
    for identifier, entry in sorted(relationships.items()):
        if not entry["zones"].issubset(zones):
            raise ValueError("Missing upstream zone result")
        if entry["zones"] != set(range(1, identifier + 1)):
            raise ValueError("Incomplete upstream zone mapping")
        selected = [zones[i] for i in entry["zones"]]
        volume = sum(row["volume_m3"] for row in selected)
        area = sum(row["area_km2"] for row in selected)
        obs = observations[entry["name"]]
        result.append(
            {
                "run_id": run_id,
                "station_id": identifier,
                "station": entry["name"],
                "year": year,
                "climate": f"CMFD_{precipitation}+TerraClimate_PET",
                "landcover": landcover,
                "natural_yield_m3": volume,
                "natural_yield_1e8_m3": volume / 1e8,
                "yield_depth_mm": volume / (area * 1e6) * 1000,
                "mean_flow_m3_s": volume / (366 if calendar.isleap(year) else 365) / 86400,
                "observed_1e8_m3": obs["value"],
                "observation_raw": obs["raw"],
                "observation_status": obs["status"],
                "area_km2": area,
                "valid_area_km2": sum(row["valid_area_km2"] for row in selected),
                "human_adjusted_m3": None,
                "human_adjustment_status": "not_available",
                "quality": "engineering_trial;uncalibrated;regulated_observations",
            }
        )
    if len(result) != limit:
        raise ValueError("Incomplete station totals")
    return result


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
