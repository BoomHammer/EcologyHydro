"""Compare identical CMFD monthly subsets using full, windowed and cached reads."""

import json
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

from ecologyhydro.config import project_root


def main():
    root = project_root()
    report = json.loads((root / "project/cache/m2/latest.json").read_text(encoding="utf-8"))
    source = next((root / report["recipe"]["climate_directory"]).glob("prec_*.nc"))
    crop = Path(report["native"]["monthly_prec"])
    records = []
    with Dataset(source) as original, Dataset(crop) as subset:
        lon, lat = original["lon"][:], original["lat"][:]
        xs = np.flatnonzero(np.isin(lon, subset["lon"][:]))
        ys = np.flatnonzero(np.isin(lat, subset["lat"][:]))
        x, y = slice(xs[0], xs[-1] + 1), slice(ys[0], ys[-1] + 1)
        axis = original["time"]
        dates = num2date(axis[:], axis.units, calendar=getattr(axis, "calendar", "standard"))
        indices = [i for i, date in enumerate(dates) if 2019 <= date.year <= 2023]
        for repetition in range(3):
            for method in ("full_then_subset", "source_window", "cached_crop"):
                started = time.perf_counter()
                checksum = 0.0
                for index in indices:
                    if method == "full_then_subset":
                        data = original["prec"][index][y, x]
                    elif method == "source_window":
                        data = original["prec"][index, y, x]
                    else:
                        data = subset["prec"][index]
                    checksum += float(np.ma.sum(data, dtype=np.float64))
                records.append(
                    {
                        "repeat": repetition,
                        "method": method,
                        "seconds": time.perf_counter() - started,
                        "checksum": checksum,
                    }
                )
    if not np.allclose([row["checksum"] for row in records], records[0]["checksum"], rtol=1e-12):
        raise RuntimeError("Cropping changed source values")
    result = {
        "scope": "60 monthly precipitation reads; OS/file caches may be warm; not AWY runtime",
        "records": records,
        "median_seconds": {
            method: float(np.median([r["seconds"] for r in records if r["method"] == method]))
            for method in {r["method"] for r in records}
        },
    }
    output = root / "project/cache/m2/crop_benchmark.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
