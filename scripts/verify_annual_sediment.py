"""Verify one annual virtual product end-to-end against station totals and R inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

from ecologyhydro.simulation import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    gdal.UseExceptions()
    folder = args.run / "rasters"
    sums = np.zeros(256, dtype=np.float64)
    counts = np.zeros(256, dtype=np.int64)
    r_error = 0.0
    domain_counts = counts.copy()
    with (
        gdal.Open(str(folder / "zones.vrt")) as zones,
        gdal.Open(str(folder / f"sed_export_{args.year}.vrt")) as export,
        gdal.Open(str(folder / f"R_{args.year}.vrt")) as annual_r,
        gdal.Open(str(folder / "base_r_nan.vrt")) as base_r,
        gdal.Open(str(folder / f"factor_{args.year}.vrt")) as factor,
    ):
        pixel_ha = abs(zones.GetGeoTransform()[1] * zones.GetGeoTransform()[5]) / 1e4
        for y in range(0, zones.RasterYSize, 128):
            window = (0, y, zones.RasterXSize, min(128, zones.RasterYSize - y))
            ids = zones.ReadAsArray(*window).astype(int)
            domain_counts += np.bincount(ids.ravel(), minlength=256)
            values = export.ReadAsArray(*window)
            valid = (ids > 0) & np.isfinite(values)
            if np.any(values[valid] < 0):
                raise ValueError("Negative export")
            sums += np.bincount(ids[valid], weights=values[valid], minlength=256) * pixel_ha
            counts += np.bincount(ids[valid], minlength=256)
            expected = base_r.ReadAsArray(*window) * factor.ReadAsArray(*window)
            actual = annual_r.ReadAsArray(*window)
            np.testing.assert_array_equal(np.isfinite(expected), np.isfinite(actual))
            valid_r = np.isfinite(expected)
            r_error = max(
                r_error, float(np.max(np.abs(actual[valid_r] - expected[valid_r]), initial=0))
            )
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    routing = Path(manifest.get("routing_dir", "project/repairs/closed_routing_v3"))
    membership = pd.read_csv(routing / "zone_station.csv")
    for row in membership.drop_duplicates("zone_id").itertuples():
        if domain_counts[row.zone_id] != row.zone_pixels:
            raise ValueError("Cropped grid dropped reporting-domain pixels")
    table = pd.read_csv(args.run / "station_annual_sediment.csv")
    table = table[table.year == args.year].set_index("station_id")
    relative = []
    for station_id, group in membership.groupby("station_id"):
        ids = group.zone_id.to_numpy()
        row = table.loc[station_id]
        if counts[ids].sum() != row.valid_hillslope_pixels:
            raise ValueError("Virtual export nodata differs from reported coverage")
        relative.append(abs(sums[ids].sum() / row.hillslope_delivery_t_yr - 1))
    if max(relative) > 1e-6 or r_error > 1e-4:
        raise ValueError("Virtual products do not reproduce reported values")
    audit = {
        "year": args.year,
        "status": "passed",
        "domain_pixels": int(domain_counts[1:].sum()),
        "max_station_relative_difference": max(relative),
        "max_r_absolute_difference": r_error,
        "scope": "complete representative-year VRT products, masks, and all station totals",
        "station_count": len(table),
    }
    write_json(args.run / "virtual_product_check.json", audit)
    print(audit)


if __name__ == "__main__":
    main()
