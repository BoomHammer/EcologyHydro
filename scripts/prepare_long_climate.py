"""Extend the fixed TerraClimate V1.1 forcing to 2013--2018, preserving old files."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from netCDF4 import Dataset, num2date
from osgeo import gdal

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import NODATA, OPTIONS, write_raster


def fetch(variable, year, output):
    path = output / f"terraclimate_{variable}_{year}.nc"
    if path.exists():
        record = read_json(path.with_suffix(".json"))
        if fingerprint(path)["sha256"] != record["file"]["sha256"]:
            raise ValueError("Provider subset checksum changed")
        return variable, year, path
    query = urlencode(
        {
            "var": variable,
            "north": 42.5,
            "south": 31.5,
            "west": 95,
            "east": 120,
            "horizStride": 1,
            "time_start": f"{year}-01-01T00:00:00Z",
            "time_end": f"{year}-12-31T23:59:59Z",
            "timeStride": 1,
            "accept": "netcdf",
        }
    )
    url = (
        "https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/"
        f"agg_terraclimate_{variable}_1950_CurrentYear_GLOBE.nc?{query}"
    )
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response:
                content = response.read(30 * 1024**2 + 1)
            if len(content) > 30 * 1024**2 or not content.startswith((b"CDF", b"\x89HDF")):
                raise ValueError("Invalid bounded NetCDF response")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
            write_json(
                path.with_suffix(".json"),
                {
                    "url": url,
                    "file": fingerprint(path),
                    "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            print("Downloaded", variable, year, flush=True)
            return variable, year, path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError("Unreachable")


def monthly_data(path, variable, year):
    with Dataset(path) as ds:
        data, t = ds[variable], ds["time"]
        dates = num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
        if ds.getncattr("version") != "V1.1" or data.units != "mm":
            raise ValueError("Provider version or units changed")
        if [(d.year, d.month) for d in dates] != [(year, m) for m in range(1, 13)]:
            raise ValueError("Incomplete or repeated provider months")
        values = np.ma.asarray(data[:], dtype=float).filled(np.nan)
        lat, lon = np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:])
        if lat[0] < lat[-1]:
            lat, values = lat[::-1], values[:, ::-1]
        if lon[0] > lon[-1]:
            lon, values = lon[::-1], values[:, :, ::-1]
        gt = (
            lon[0] - abs(lon[1] - lon[0]) / 2,
            abs(lon[1] - lon[0]),
            0,
            lat[0] + abs(lat[1] - lat[0]) / 2,
            0,
            -abs(lat[1] - lat[0]),
        )
        metadata = {k: str(data.getncattr(k)) for k in data.ncattrs()}
    if np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Negative monthly water depth")
    return values, gt, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/repairs/long_climate_v1")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root.resolve() / "project/repairs"):
        raise ValueError("Output must be in project/repairs")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Completed extension already exists; choose a new output")
    started = time.perf_counter()
    # Download concurrently, but decode NetCDF sequentially (netCDF C is not thread safe).
    jobs = [(v, y) for y in range(2013, 2019) for v in ("ppt", "pet")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        downloads = list(pool.map(lambda job: fetch(*job, output), jobs))
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    grid = read_json(root / "project/repairs/soil_routing/m2_index.json")["grid"]
    records = []
    for variable, year, path in downloads:
        values, gt, metadata = monthly_data(path, variable, year)
        total = values.sum(axis=0)
        native, aligned = (
            output / f"{variable}_{year}.tif",
            output / f"{variable}_{year}_aligned.tif",
        )
        write_raster(native, np.where(np.isfinite(total), total, NODATA), gt, "EPSG:4326")
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
        records.append(
            {
                "variable": variable,
                "year": year,
                "raw": fingerprint(path),
                "aligned": fingerprint(aligned),
                "metadata": metadata,
            }
        )
        print("Prepared", variable, year, flush=True)
    write_json(
        output / "summary.json",
        {
            "version": "V1.1",
            "years": list(range(2013, 2019)),
            "records": records,
            "elapsed_seconds": time.perf_counter() - started,
            "method": "Decode packed monthly mm once; sum exactly 12 months; same model grid",
            "reference": "https://www.climatologylab.org/terraclimate.html",
            "script": fingerprint(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
