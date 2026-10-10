"""Conditional rainfall reconstruction from independently supplied regional depths."""

from pathlib import Path

import numpy as np
from calibrate_regional_water import source_paths
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.reference_checks import REGIONS
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import OPTIONS, windows
from ecologyhydro.water_balance import read_csv


def main():
    root = project_root()
    output = root / "project/diagnostics/reference_precipitation_v1"
    output.mkdir(parents=True, exist_ok=False)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    rain_path = root / "data/Hydrology/降水量2018-2023.csv"
    area_path = root / "data/Hydrology/水资源二级区面积.csv"
    rain = {r["水资源二级区"]: r for r in read_csv(rain_path)}
    area = {r["水资源二级区"]: float(r["计算面积(万平方千米)"]) for r in read_csv(area_path)}
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    zone_path = root / "project/diagnostics/gonghe_connectivity_v1/zones.tif"
    rows, sources = [], [rain_path, area_path, zone_path, Path(__file__)]
    for year in range(2019, 2024):
        path = Path(source_paths(root, index, "fine", year)["p"])
        sources.append(path)
        with gdal.Open(str(zone_path)) as zones, gdal.Open(str(path)) as p:
            counts, totals = np.zeros(11), np.zeros(11)
            for window in windows(zones):
                z, value = zones.ReadAsArray(*window), p.ReadAsArray(*window)
                valid = z > 0
                if (value[valid] <= 0).any() or not np.isfinite(value[valid]).all():
                    raise ValueError("Invalid annual precipitation")
                counts += np.bincount(z[valid].astype(int) - 1, minlength=11)
                totals += np.bincount(z[valid].astype(int) - 1, weights=value[valid], minlength=11)
            factors = np.ones(12)
            for region, start, stop in REGIONS[:6]:
                mean = totals[start:stop].sum() / counts[start:stop].sum()
                reference = float(rain[region][str(year)]) * 10 / area[region]
                factor = reference / mean
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Invalid regional reference")
                factors[start + 1 : stop + 1] = factor
                rows.append(
                    {
                        "year": year,
                        "region": region,
                        "model_mean_mm": mean,
                        "reference_mean_mm": reference,
                        "factor": factor,
                        "same_boundary": False,
                    }
                )
            with gdal.GetDriverByName("GTiff").CreateCopy(
                str(output / f"precipitation_{year}.tif"), p, options=OPTIONS
            ) as result:
                for window in windows(zones):
                    z, value = zones.ReadAsArray(*window), p.ReadAsArray(*window)
                    valid = z > 0
                    value[valid] *= factors[z[valid].astype(int)]
                    result.GetRasterBand(1).WriteArray(value, window[0], window[1])
    write_csv(output / "factors.csv", rows)
    write_json(
        output / "summary.json",
        {
            "purpose": "conditional retrospective rainfall reconstruction; not future forecast",
            "method": "match regional mean rainfall depth, preserve within-region spatial pattern",
            "used_runoff_to_fit_factors": False,
            "reference_mm": "tabulated volume in 1e8 m3 times 10 / area in 1e4 km2",
            "regions": "first six station-proxy regions only; downstream unchanged",
            "same_boundary": False,
            "limitation": "Proxy regional boundaries; exact spatial allocation unvalidated",
            "2023": "uses 2023 precipitation observations as meteorological input, not runoff",
            "future_use": "requires a separate bias-adjustment design for climate scenarios",
            "sources": [fingerprint(path) for path in sources],
        },
    )
    print(rows, flush=True)


if __name__ == "__main__":
    main()
