"""Compare provider reference subsets with local inputs on the repaired upper basin."""

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.spatial import NODATA, OPTIONS, windows, write_raster


def main():
    gdal.UseExceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-basin", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    root = project_root()
    name = "climate_reference_full" if args.full_basin else "climate_reference"
    if args.holdout:
        name += "_holdout"
    output = root / "project/repairs" / name
    index = json.loads((root / "project/cache/m2/latest.json").read_text(encoding="utf-8"))
    records, metadata = [], []
    with gdal.Open(str(Path(index["routing"]["partitions"]) / "zones.tif")) as zones:
        zone_gt = zones.GetGeoTransform()
        pixel_area = abs(zone_gt[1] * zone_gt[5] - zone_gt[2] * zone_gt[4])
        for path in sorted((output / "provider").glob("*.nc")):
            variable, year = path.stem.split("_")[-2:]
            year = int(year)
            with Dataset(path) as ds:
                data = ds[variable]
                if data.units != "mm" or data.dimensions != ("time", "lat", "lon"):
                    raise ValueError("Unsupported provider units or dimensions")
                if ds.getncattr("version") != "V1.1":
                    raise ValueError("Provider version changed; review before combining years")
                metadata.append(
                    {
                        "file": str(path),
                        "global": {k: str(ds.getncattr(k)) for k in ds.ncattrs()},
                        "variable": {k: str(data.getncattr(k)) for k in data.ncattrs()},
                    }
                )
                t = ds["time"]
                dates = num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
                if [(d.year, d.month) for d in dates] != [(year, m) for m in range(1, 13)]:
                    raise ValueError("Provider subset has unexpected month coverage")
                values = np.ma.asarray(data[:], dtype=float).filled(np.nan)
                if np.any(values[np.isfinite(values)] < 0):
                    raise ValueError("Negative provider water depth")
                lat, lon = np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:])
                if lat[0] < lat[-1]:
                    values, lat = values[:, ::-1], lat[::-1]
                if lon[0] > lon[-1]:
                    values, lon = values[:, :, ::-1], lon[::-1]
                gt = (
                    lon[0] - abs(lon[1] - lon[0]) / 2,
                    abs(lon[1] - lon[0]),
                    0,
                    lat[0] + abs(lat[1] - lat[0]) / 2,
                    0,
                    -abs(lat[1] - lat[0]),
                )
                total = np.sum(values, axis=0)
            native = output / f"provider_{variable}_{year}.tif"
            aligned = output / f"provider_{variable}_{year}_aligned.tif"
            write_raster(native, np.where(np.isfinite(total), total, NODATA), gt, "EPSG:4326")
            grid = index["grid"]
            with gdal.Warp(
                str(aligned),
                str(native),
                dstSRS=grid["crs"],
                outputBounds=grid["bounds"],
                width=grid["width"],
                height=grid["height"],
                resampleAlg="bilinear",
                errorThreshold=0,
                dstNodata=NODATA,
                outputType=gdal.GDT_Float32,
                creationOptions=OPTIONS,
            ):
                pass
            compare = {"pet": "pet", "ppt": "prec"}.get(variable)
            paths = {"provider": str(aligned)}
            if compare:
                paths["local"] = index["aligned"][f"{compare}_{year}"]
            for label, raster in paths.items():
                summed = count = 0
                with gdal.Open(raster) as src:
                    for window in windows(zones):
                        inside = zones.ReadAsArray(*window) == 1
                        if not inside.any():
                            continue
                        selected = src.ReadAsArray(*window)[inside]
                        if np.any(selected < 0) or not np.isfinite(selected).all():
                            raise ValueError(
                                "Provider reference fails complete Tangnaihai coverage"
                            )
                        summed += float(selected.sum(dtype=float))
                        count += len(selected)
                records.append(
                    {
                        "variable": variable,
                        "year": year,
                        "source": label,
                        "mean_mm": summed / count,
                        "volume_1e8_m3": summed * pixel_area / 1e11,
                    }
                )
    write_csv(output / "provider_comparison.csv", records)
    (output / "provider_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "provider_check_manifest.json").write_text(
        json.dumps(
            {
                "sources": [fingerprint(p) for p in sorted((output / "provider").glob("*.nc"))],
                "implementation": fingerprint(Path(__file__)),
                "grid": index["grid"],
                "partitions": index["routing"]["partitions"],
                "method": "decode packed monthly mm once; 12 dated months; exact bilinear warp",
                "provider_version": "V1.1",
                "local_pet_version": "unverified",
                "comparison_is_independent_truth": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))
    print(f"Full version and variable metadata saved: {output / 'provider_metadata.json'}")


if __name__ == "__main__":
    main()
