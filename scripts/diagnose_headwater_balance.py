"""Headwater monthly process audit with exact pixel-centre bilinear area weights."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from osgeo import gdal, osr
from prepare_long_climate import monthly_data

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.long_record import YEARS, load_accounts
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import crs, windows

VARIABLES = ("aet", "q", "soil", "swe")


def fetch(variable, output):
    path = output / f"{variable}_201212_202212.nc"
    if path.exists():
        if fingerprint(path)["sha256"] != read_json(path.with_suffix(".json"))["file"]["sha256"]:
            raise ValueError("Cached source changed")
        return path
    query = urlencode(
        dict(
            var=variable,
            north=39,
            south=31.5,
            west=95,
            east=105,
            horizStride=1,
            time_start="2012-12-01T00:00:00Z",
            time_end="2022-12-31T23:59:59Z",
            timeStride=1,
            accept="netcdf",
        )
    )
    url = (
        "https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/"
        + f"agg_terraclimate_{variable}_1950_CurrentYear_GLOBE.nc?{query}"
    )
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response:
                content = response.read(30 * 1024**2 + 1)
            if len(content) > 30 * 1024**2 or not content.startswith((b"CDF", b"\x89HDF")):
                raise ValueError("Invalid bounded provider response")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
            write_json(
                path.with_suffix(".json"),
                dict(
                    url=url,
                    file=fingerprint(path),
                    retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ),
            )
            print("Downloaded", variable, flush=True)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError("Unreachable")


def provider_series(path, variable):
    with Dataset(path) as dataset:
        v, t = dataset[variable], dataset["time"]
        dates = num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
        expected = [(2012, 12)] + [(y, m) for y in YEARS for m in range(1, 13)]
        if [(d.year, d.month) for d in dates] != expected:
            raise ValueError("Incomplete monthly states/fluxes")
        if dataset.getncattr("version") != "V1.1" or v.units != "mm":
            raise ValueError("Wrong version or units")
        metadata = {k: str(v.getncattr(k)) for k in v.ncattrs()}
        values = np.ma.asarray(v[:], dtype=float).filled(np.nan)
        lat, lon = np.asarray(dataset["lat"][:]), np.asarray(dataset["lon"][:])
        if lat[0] < lat[-1]:
            lat, values = lat[::-1], values[:, ::-1]
        if lon[0] > lon[-1]:
            lon, values = lon[::-1], values[:, :, ::-1]
        dx, dy = float(lon[1] - lon[0]), float(lat[0] - lat[1])
        gt = (float(lon[0] - dx / 2), dx, 0, float(lat[0] + dy / 2), 0, -dy)
    return values, gt, metadata


def spatial_weights(zone_path, gt, shape):
    """All upper-basin model cells contribute exactly their equal-area pixel area."""
    height, width = shape
    weights = np.zeros((2, height * width))
    with gdal.Open(str(zone_path)) as zones:
        grid = zones.GetGeoTransform()
        transform = osr.CoordinateTransformation(crs(zones.GetProjection()), crs("EPSG:4326"))
        volume = abs(grid[1] * grid[5] - grid[2] * grid[4]) / 1e11
        for xoff, yoff, nx, ny in windows(zones):
            z = zones.ReadAsArray(xoff, yoff, nx, ny)
            yy, xx = np.where((z > 0) & (z <= 3))
            if not len(xx):
                continue
            group = (z[yy, xx] == 3).astype(int)
            lonlat = np.asarray(
                transform.TransformPoints(
                    np.column_stack(
                        [
                            grid[0] + (xx + xoff + 0.5) * grid[1],
                            grid[3] + (yy + yoff + 0.5) * grid[5],
                        ]
                    )
                )
            )
            col = (lonlat[:, 0] - gt[0]) / gt[1] - 0.5
            row = (lonlat[:, 1] - gt[3]) / gt[5] - 0.5
            c, r = np.floor(col).astype(int), np.floor(row).astype(int)
            if np.any((c < 0) | (r < 0) | (c + 1 >= width) | (r + 1 >= height)):
                raise ValueError("Provider subset does not cover headwater pixels")
            dc, dr = col - c, row - r
            for di, dj, factor in (
                (0, 0, (1 - dr) * (1 - dc)),
                (0, 1, (1 - dr) * dc),
                (1, 0, dr * (1 - dc)),
                (1, 1, dr * dc),
            ):
                index = group * height * width + (r + di) * width + c + dj
                weights += np.bincount(
                    index, weights=factor * volume, minlength=2 * height * width
                ).reshape(2, -1)
    return weights


def aggregate(values, weights):
    selected = weights.sum(axis=0) > 0
    data = values.reshape(len(values), -1)[:, selected]
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Invalid contributing monthly cells")
    return data @ weights[:, selected].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/diagnostics/headwater_balance_v1")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "project/diagnostics"):
        raise ValueError("Output must be in diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Completed diagnostic exists")
    configure_threads(load_config().resources)
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    start = time.perf_counter()
    zone_path = root / "project/repairs/closed_routing_v3/zones.tif"
    write_json(
        output / "protocol.json",
        {
            "scope": "Guide and Guide-Lanzhou; diagnose main upstream annual mismatch",
            "years": YEARS,
            "prior_state": "December 2012 only; no invented 2012 hydrology",
            "fit_parameters": False,
            "no_new_2023_prediction": True,
            "source": "TerraClimate V1.1 monthly aet/q/soil/swe plus existing ppt/pet",
            "method": "Bilinear interpolation at every model pixel centre; equal-area weights",
            "closure": "P-AET-Q-dSoil-dSnow; also minus dQ for V1.1 runoff carryover",
            "limits": (
                "Modeled same-forcing benchmark, not independent observations "
                "or validated replacement for AWY"
            ),
            "references": [
                "https://www.climatologylab.org/terraclimate.html",
                "https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/annual_water_yield.html",
            ],
        },
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda v: fetch(v, output), VARIABLES))
    cache, series, metadata, sources = {}, {}, {}, [fingerprint(zone_path)]

    def reduce(values, gt):
        key = (tuple(gt), values.shape[1:])
        if key not in cache:
            cache[key] = spatial_weights(zone_path, gt, values.shape[1:])
            print("Computed full-pixel spatial weights", values.shape[1:], flush=True)
        return aggregate(values, cache[key])

    for variable, path in zip(VARIABLES, paths, strict=True):
        values, gt, metadata[variable] = provider_series(path, variable)
        series[variable] = reduce(values, gt)
        sources.append(fingerprint(path))
    for variable in ("ppt", "pet"):
        rows = []
        for year in YEARS:
            folder = root / "project/repairs"
            path = (
                folder / "long_climate_v1" / f"terraclimate_{variable}_{year}.nc"
                if year <= 2018
                else folder
                / "climate_reference_full/provider"
                / f"terraclimate_current_{variable}_{year}.nc"
            )
            values, gt, _ = monthly_data(path, variable, year)
            rows.extend(reduce(values, gt))
            sources.append(fingerprint(path))
        series[variable] = np.vstack([np.full((1, 2), np.nan), rows])
    monthly = []
    for index in range(1, 121):
        year, month = 2013 + (index - 1) // 12, (index - 1) % 12 + 1
        for region in range(2):
            delta_s = series["soil"][index, region] - series["soil"][index - 1, region]
            delta_w = series["swe"][index, region] - series["swe"][index - 1, region]
            delta_q = series["q"][index, region] - series["q"][index - 1, region]
            closure = (
                series["ppt"][index, region]
                - series["aet"][index, region]
                - series["q"][index, region]
                - delta_s
                - delta_w
            )
            monthly.append(
                dict(
                    year=year,
                    month=month,
                    region_id=region + 1,
                    **{v: float(a[index, region]) for v, a in series.items()},
                    delta_soil=delta_s,
                    delta_snow=delta_w,
                    delta_runoff_carry=delta_q,
                    closure_without_carry=closure,
                    closure_with_carry=closure - delta_q,
                )
            )
    frame = pd.DataFrame(monthly)
    frame.to_csv(output / "monthly_balance.csv", index=False)
    fluxes = [
        "ppt",
        "pet",
        "aet",
        "q",
        "delta_soil",
        "delta_snow",
        "delta_runoff_carry",
        "closure_without_carry",
        "closure_with_carry",
    ]
    annual = frame.groupby(["year", "region_id"])[fluxes].sum().reset_index()
    annual.to_csv(output / "annual_local_balance.csv", index=False)
    target, consumption, storage, _, eligible = load_accounts(root)
    original = pd.read_csv(root / "project/calibration/long_record_v1/climate_regions_fine.csv")
    checks, records = [], []
    for yi, year in enumerate(YEARS):
        cumulative = annual[annual.year == year].sort_values("region_id")[fluxes].cumsum()
        for si, station in enumerate(("贵得", "兰州")):
            record = dict(year=year, station=station, **cumulative.iloc[si].to_dict())
            adjustment = np.sum(consumption[yi, : si + 1] + storage[yi, : si + 1])
            record.update(
                observed=target[yi, si],
                known_adjustment=adjustment,
                eligible=bool(eligible[yi, si]),
                provider_conditional_q=record["q"] - adjustment,
            )
            records.append(record)
            ref = original[(original.year == year) & (original.region_id == si + 1)].iloc[0]
            amount = float(annual[(annual.year == year) & (annual.region_id == si + 1)].ppt.iloc[0])
            reference = ref.precipitation_mm * ref.area_km2 / 1e5
            checks.append(abs(amount / reference - 1))
    pd.DataFrame(records).to_csv(output / "annual_cumulative_balance.csv", index=False)
    if max(checks) > 0.001:
        raise ValueError("Monthly aggregation differs from annual grid by >0.1%")
    write_json(
        output / "summary.json",
        {
            "elapsed_seconds": time.perf_counter() - start,
            "metadata": metadata,
            "max_precipitation_difference_fraction_from_annual_grid": max(checks),
            "max_monthly_closure_without_carry_1e8_m3": float(
                frame.closure_without_carry.abs().max()
            ),
            "max_monthly_closure_with_carry_1e8_m3": float(frame.closure_with_carry.abs().max()),
            "sources": sources
            + [fingerprint(Path(__file__)), fingerprint(output / "protocol.json")],
            "units": "All water quantities in 1e8 m3; monthly soil/swe are end states, not sums",
            "independent_measurement": False,
            "production_changed": False,
        },
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
