"""Inventory the new observations, missing-account propagation and regional closure."""

import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.long_record import ENDPOINTS, STATIONS, load_accounts, number, table
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import REGIONS, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/diagnostics/long_record_v1")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root.resolve() / "project/diagnostics"):
        raise ValueError("Output must be under project/diagnostics")
    output.mkdir(exist_ok=False)
    folder = root / "data/Hydrology"
    missing, overlap, fields = [], [], []
    sources = sorted(folder.glob("*2013-2023.csv"))
    for path in sources:
        rows = read_csv(path)
        label = "测站" if path.name.startswith("实测") else "水资源二级区"
        old_path = path.with_name(path.name.replace("2013", "2018"))
        old = table(old_path, label)
        for row in rows:
            for year in range(2013, 2024):
                value = number(row.get(str(year)))
                record = {"file": path.name, "label": row[label], "year": year}
                if not np.isfinite(value):
                    missing.append(record)
                fields.append({**record, "value": value if np.isfinite(value) else None})
                if year >= 2018:
                    previous = number(old[row[label]].get(str(year)))
                    equal = (np.isnan(value) and np.isnan(previous)) or value == previous
                    if not equal:
                        overlap.append({**record, "old": previous, "new": value})
    availability, residuals = [], []
    for version in ("legacy", "sector_2022"):
        years = list(range(2013, 2024))
        target, c, s, adjustment, eligible = load_accounts(root, version, years)
        for yi, year in enumerate(years):
            for si, station in enumerate(STATIONS):
                availability.append(
                    {
                        "account": version,
                        "year": year,
                        "station": station,
                        "observation_present": bool(np.isfinite(target[yi, si])),
                        "cumulative_account_present": bool(np.isfinite(adjustment[yi, si])),
                        "eligible": bool(eligible[yi, si]),
                    }
                )
                upstream = target[yi, si - 1] if si else 0.0
                restored = target[yi, si] - upstream + c[yi, si] + s[yi, si]
                residuals.append(
                    {
                        "account": version,
                        "year": year,
                        "region": REGIONS[si],
                        "upstream_proxy": STATIONS[si - 1] if si else "headwaters",
                        "downstream_proxy": station,
                        "partially_restored_increment": float(restored)
                        if np.isfinite(restored)
                        else None,
                        "negative_increment": bool(restored < 0) if np.isfinite(restored) else None,
                    }
                )
    gdal.UseExceptions()
    with gdal.Open(str(root / "project/repairs/closed_routing_v3/zones.tif")) as zones:
        count = np.bincount(zones.ReadAsArray().ravel(), minlength=12)[1:]
        area = count * abs(zones.GetGeoTransform()[1] * zones.GetGeoTransform()[5]) / 1e6
    cumulative = area.cumsum()[ENDPOINTS]
    proxy_area = np.diff(np.r_[0, cumulative])
    official_area = table(folder / "水资源二级区面积.csv", "水资源二级区")
    mapping = [
        {
            "region": r,
            "endpoint_proxy": STATIONS[i],
            "proxy_area_km2": proxy_area[i],
            "statistical_area_km2": float(official_area[r]["计算面积(万平方千米)"]) * 1e4,
            "status": "station_proxy_not_official_polygon",
            "caveat": "Guide is not Longyangxia"
            if i == 0
            else "Lijin is not the river mouth; below-Huayuankou accounts are a conditional proxy"
            if i == 6
            else "Derived from nested gauge catchments",
        }
        for i, r in enumerate(REGIONS[:7])
    ]
    write_csv(output / "missing.csv", missing)
    write_csv(output / "availability.csv", availability)
    write_csv(output / "reach_balance.csv", residuals)
    write_csv(output / "region_mapping.csv", mapping)
    write_csv(output / "normalized_fields.csv", fields)
    write_json(
        output / "summary.json",
        {
            "years_in_new_files": list(range(2013, 2024)),
            "year_2012_not_in_new_files": True,
            "missing_fields": len(missing),
            "overlap_changes_2018_2023": overlap,
            "missing_policy": "No zero fill or interpolation; propagate missing upstream accounts",
            "user_boundary_policy": "No official polygons; use gauges and routed catchments",
            "three_region_correction": "Toudaoguai/Huayuankou replace Lanzhou/Longmen breaks",
            "sources": [fingerprint(p) for p in [*sources, Path(__file__)]],
            "references": ["https://slsy.nhri.cn/cn/article/doi/10.12170/20210923001"],
        },
    )
    print("Missing fields:", len(missing), "changed overlapping cells:", len(overlap), flush=True)


if __name__ == "__main__":
    main()
