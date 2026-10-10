"""Retrieve December modeled storage states; aggregate annual changes, never annual sums."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from netCDF4 import Dataset, num2date
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import NODATA, OPTIONS, windows, write_raster


def fetch(variable, output):
    query = urlencode(
        {
            "var": variable,
            "north": 42.5,
            "south": 31.5,
            "west": 95,
            "east": 120,
            "horizStride": 1,
            "time_start": "2018-12-01T00:00:00Z",
            "time_end": "2023-12-31T23:59:59Z",
            "timeStride": 12,
            "accept": "netcdf",
        }
    )
    url = (
        "https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/"
        f"agg_terraclimate_{variable}_1950_CurrentYear_GLOBE.nc?{query}"
    )
    path = output / f"{variable}_december_2018_2023.nc"
    with urlopen(url, timeout=45) as response:
        data = response.read(30 * 1024**2 + 1)
    if len(data) > 30 * 1024**2 or not data.startswith((b"CDF", b"\x89HDF")):
        raise ValueError("Invalid bounded NetCDF response")
    path.write_bytes(data)
    write_json(
        path.with_suffix(".json"),
        {
            "url": url,
            "file": fingerprint(path),
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return variable, path


def main():
    root = project_root()
    output = root / "project/diagnostics/storage_change_v1"
    output.mkdir(parents=True, exist_ok=False)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = dict(pool.map(lambda variable: fetch(variable, output), ("soil", "swe")))
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    domains = {
        "original": Path(index["routing"]["partitions"]) / "zones.tif",
        "gonghe_excluded": root / "project/diagnostics/gonghe_connectivity_v1/zones.tif",
    }
    rows, metadata = [], {}
    for variable, path in sources.items():
        with Dataset(path) as ds:
            v, t = ds[variable], ds["time"]
            dates = num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
            if [(d.year, d.month) for d in dates] != [(y, 12) for y in range(2018, 2024)]:
                raise ValueError("Expected December states for 2018--2023")
            if ds.getncattr("version") != "V1.1" or v.units != "mm":
                raise ValueError("Unexpected provider version/units")
            metadata[variable] = {k: str(v.getncattr(k)) for k in v.ncattrs()}
            values = np.ma.asarray(v[:], dtype=float).filled(np.nan)
            lat, lon = ds["lat"][:], ds["lon"][:]
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
        for yi, year in enumerate(range(2019, 2024), 1):
            change = values[yi] - values[yi - 1]
            native, aligned = (
                output / f"{variable}_{year}.tif",
                output / f"{variable}_{year}_aligned.tif",
            )
            write_raster(native, np.where(np.isfinite(change), change, NODATA), gt, "EPSG:4326")
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
            for name, zone_path in domains.items():
                with gdal.Open(str(zone_path)) as zones, gdal.Open(str(aligned)) as raster:
                    totals = np.zeros(11)
                    area = abs(zones.GetGeoTransform()[1] * zones.GetGeoTransform()[5]) / 1e11
                    for window in windows(zones):
                        z, s = zones.ReadAsArray(*window), raster.ReadAsArray(*window)
                        valid = z > 0
                        if np.any(s[valid] == NODATA) or not np.isfinite(s[valid]).all():
                            raise ValueError("Missing storage state within catchments")
                        totals += np.bincount(
                            z[valid].astype(int) - 1, weights=s[valid] * area, minlength=11
                        )
                    for zone, amount in enumerate(totals, 1):
                        rows.append(
                            {
                                "domain": name,
                                "year": year,
                                "zone": zone,
                                "variable": variable,
                                "delta_storage_1e8_m3": amount,
                            }
                        )
        print(variable, "aggregated", flush=True)
    write_csv(output / "storage_changes.csv", rows)
    write_json(
        output / "summary.json",
        {
            "source": "https://www.climatologylab.org/terraclimate.html",
            "method": "consecutive December soil/snow state differences; coefficient fixed at one",
            "limitation": "Modeled proxy; correlated forcing, static vegetation, no groundwater",
            "independent_measurement": False,
            "metadata": metadata,
            "sources": [fingerprint(p) for p in sources.values()] + [fingerprint(Path(__file__))],
        },
    )


if __name__ == "__main__":
    main()
