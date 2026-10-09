"""Diagnostic monthly-mean PM approximation; never replaces production ET0."""

import calendar
import csv
import json
import time
from contextlib import ExitStack
from pathlib import Path

from ecologyhydro.cache import ArtifactCache
from ecologyhydro.config import load_config, project_root
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json

UNITS = {
    "temp": "K",
    "pres": "Pa",
    "shum": "kg kg-1",
    "wind": "m s-1",
    "srad": "W m-2",
    "lrad": "W m-2",
    "prec": "kg m-2 s-1",
}


def approximate_pm(temp, pressure, humidity, wind, shortwave, longwave, height):
    """Return mm/day and diagnostics with G=0 and surface temperature=air temperature."""
    import numpy as np

    if height not in (2, 10):
        raise ValueError("Only the declared 2 m / 10 m wind hypotheses are supported")
    temp, pressure, humidity, wind, shortwave, longwave = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (temp, pressure, humidity, wind, shortwave, longwave)]
    )
    celsius = temp - 273.15
    es = 0.6108 * np.exp(17.27 * celsius / (celsius + 237.3))
    ea = humidity * (pressure / 1000) / (0.622 + 0.378 * humidity)
    vpd = np.maximum(es - ea, 0)
    delta = 4098 * es / (celsius + 237.3) ** 2
    gamma = 0.000665 * pressure / 1000
    u2 = wind if height == 2 else wind * 4.87 / np.log(67.8 * height - 5.42)
    # Reference albedo; emissivity and surface temperature are explicit approximations.
    radiation = (1 - 0.23) * shortwave + 0.98 * (longwave - 5.670374419e-8 * temp**4)
    radiation *= 0.0864
    raw = (0.408 * delta * radiation + gamma * 900 / (celsius + 273) * u2 * vpd) / (
        delta + gamma * (1 + 0.34 * u2)
    )
    return np.maximum(raw, 0), {"negative_et": raw < 0, "supersaturated": ea > es}


def read_months(path, name, year):
    import numpy as np
    from netCDF4 import Dataset, num2date

    with Dataset(path) as ds:
        var = ds[name]
        if var.dimensions != ("time", "lat", "lon") or var.units != UNITS[name]:
            raise ValueError(f"Unsupported metadata: {name}, {var.dimensions}, {var.units}")
        t = ds["time"]
        cal = getattr(t, "calendar", "standard")
        if cal not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError(f"Unsupported calendar: {cal}")
        dates = num2date(t[:], t.units, calendar=cal)
        selected = sorted((d.month, i) for i, d in enumerate(dates) if d.year == year)
        if [m for m, _ in selected] != list(range(1, 13)):
            raise ValueError(f"Missing/duplicate months in {name}, {year}")
        values = np.ma.asarray(var[[i for _, i in selected]], dtype=float).filled(np.nan)
        return values, np.asarray(ds["lon"][:]), np.asarray(ds["lat"][:])


def native_diagnostics(paths, year, output):
    import numpy as np

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.spatial import NODATA, write_raster

    arrays, lon, lat = {}, None, None
    for name, path in paths.items():
        array, x, y = read_months(path, name, year)
        if lon is not None and (not np.array_equal(lon, x) or not np.array_equal(lat, y)):
            raise ValueError("Climate variable grids differ")
        lon, lat, arrays[name] = x, y, array
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    valid &= (arrays["temp"] > 180) & (arrays["temp"] < 340)
    valid &= (arrays["pres"] > 10000) & (arrays["pres"] < 120000)
    valid &= (arrays["shum"] >= 0) & (arrays["shum"] < 0.1)
    for name in ("wind", "srad", "lrad", "prec"):
        valid &= arrays[name] >= 0
    annual_valid = valid.all(axis=0)
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])[:, None, None]
    annual = {"prec_recomputed": (arrays["prec"] * days * 86400).sum(axis=0)}
    records = []
    for height in (10, 2):
        et, flags = approximate_pm(
            *[arrays[k] for k in ("temp", "pres", "shum", "wind", "srad", "lrad")], height
        )
        annual[f"pm_wind{height}m"] = (et * days).sum(axis=0)
        for m in range(12):
            mask = valid[m]
            records.append(
                {
                    "year": year,
                    "month": m + 1,
                    "wind_height_hypothesis_m": height,
                    "native_bbox_valid_cells": int(mask.sum()),
                    "negative_et_clamped_cells": int((flags["negative_et"][m] & mask).sum()),
                    "supersaturated_vpd_clamped_cells": int(
                        (flags["supersaturated"][m] & mask).sum()
                    ),
                    "bbox_et0_mean_mm": float(np.mean(et[m][mask]) * days[m, 0, 0]),
                    "statistic_scope": "native_rectangle_not_watershed",
                }
            )
    dx, dy = float(np.diff(lon).mean()), float(np.diff(lat).mean())
    if not np.allclose(np.diff(lon), dx, rtol=1e-3) or not np.allclose(np.diff(lat), dy, rtol=1e-3):
        raise ValueError("Irregular climate grid")
    transform = (
        float(lon.min()) - abs(dx) / 2,
        abs(dx),
        0,
        float(lat.max()) + abs(dy) / 2,
        0,
        -abs(dy),
    )
    for name, values in annual.items():
        data = np.where(annual_valid, values, NODATA)
        if dx < 0:
            data = data[:, ::-1]
        if dy > 0:
            data = data[::-1]
        write_raster(output / f"{name}.tif", data, transform, "EPSG:4326")
    write_csv(output / "monthly_diagnostics.csv", records)
    return {
        "year": year,
        "units": "mm/year",
        "invalid_annual_native_cells": int((~annual_valid).sum()),
        "status": "approximate_comparator_not_validated_ET0",
        "G": 0,
        "surface_temperature": "monthly_air_temperature",
        "emissivity": 0.98,
        "albedo": 0.23,
        "temperature_extremes": "unavailable_mean_only",
    }


def station_statistics(index, paths, year, output):
    import numpy as np
    from osgeo import gdal

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.spatial import NODATA, windows

    partition = Path(index["routing"]["partitions"])
    with (partition / "zone_station.csv").open(encoding="utf-8-sig") as stream:
        mapping = list(csv.DictReader(stream))
    stations = {}
    for row in mapping:
        stations.setdefault(row["station"], set()).add(int(row["zone_id"]))
    grid = index["grid"]
    records = []
    with gdal.Open(str(partition / "zones.tif")) as zones:
        gt = zones.GetGeoTransform()
        area = abs(gt[1] * gt[5] - gt[2] * gt[4])
        for name, path in paths.items():
            with ExitStack() as stack:
                if name in {"pm_wind10m", "pm_wind2m", "prec_recomputed"}:
                    ds = stack.enter_context(
                        gdal.Warp(
                            "",
                            str(path),
                            format="VRT",
                            dstSRS=grid["crs"],
                            outputBounds=grid["bounds"],
                            width=grid["width"],
                            height=grid["height"],
                            resampleAlg="bilinear",
                            errorThreshold=0,
                            dstNodata=NODATA,
                            outputType=gdal.GDT_Float32,
                        )
                    )
                else:
                    ds = stack.enter_context(gdal.Open(str(path)))
                if (
                    ds.GetGeoTransform() != gt
                    or not ds.GetSpatialRef().IsSame(zones.GetSpatialRef())
                    or (ds.RasterXSize, ds.RasterYSize) != (zones.RasterXSize, zones.RasterYSize)
                ):
                    raise ValueError("Diagnostic grid mismatch")
                sums, counts = np.zeros(256), np.zeros(256)
                for block in windows(zones):
                    ids = zones.ReadAsArray(*block)
                    mask = ids > 0
                    if not mask.any():
                        continue
                    values = ds.ReadAsArray(*block)[mask].astype(float)
                    usable = ds.GetRasterBand(1).GetMaskBand().ReadAsArray(*block)[mask] > 0
                    if not usable.all() or not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError(f"Missing/invalid diagnostic data: {name}")
                    sums += np.bincount(ids[mask], weights=values, minlength=256)
                    counts += np.bincount(ids[mask], minlength=256)
                for station, zone_ids in stations.items():
                    total = sum(sums[z] for z in zone_ids)
                    count = sum(counts[z] for z in zone_ids)
                    records.append(
                        {
                            "year": year,
                            "station": station,
                            "variable": name,
                            "mean_mm": total / count,
                            "volume_1e8_m3": total * area / 1e11,
                            "area_km2": count * area / 1e6,
                        }
                    )
    write_csv(output / "stations.csv", records)
    return {"records": len(records), "method": "bilinear_to_M2_grid_then_exact_zone_sum"}


def main():
    root, started = project_root(), time.perf_counter()
    config = load_config()
    configure_threads(config.resources)
    import numpy as np
    from osgeo import gdal

    from ecologyhydro.aggregation import write_csv
    from ecologyhydro.model_inputs import verified_artifact

    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    index_path = root / "project/cache/m2/latest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    paths = {k: Path(index["native"][f"monthly_{k}"]) for k in UNITS}
    for path in paths.values():
        verified_artifact(path)
    cache = ArtifactCache(config.paths.cache / "m4/climate_review")
    artifacts, records = [], []
    implementation = [Path(__file__), Path(__file__).with_name("spatial.py")]
    partition = Path(index["routing"]["partitions"])
    for year in config.study.calibration_years:
        print(f"Preparing climate comparator {year}", flush=True)
        native = cache.build(
            "native",
            [*paths.values(), *implementation],
            {"year": year},
            lambda out, y=year: native_diagnostics(paths, y, out),
        )
        rasters = {k: native / f"{k}.tif" for k in ("pm_wind10m", "pm_wind2m", "prec_recomputed")}
        rasters.update({k: Path(index["aligned"][f"{k}_{year}"]) for k in ("pet", "prec", "bcpr")})
        original_path = Path(index["native"][f"prec_{year}"])
        verified_artifact(original_path)
        with (
            gdal.Open(str(native / "prec_recomputed.tif")) as recomputed,
            gdal.Open(str(original_path)) as original,
        ):
            if (
                recomputed.GetGeoTransform() != original.GetGeoTransform()
                or not recomputed.GetSpatialRef().IsSame(original.GetSpatialRef())
            ):
                raise ValueError("Recomputed native precipitation grid differs")
            a, b = recomputed.ReadAsArray(), original.ReadAsArray()
            mask_a = recomputed.GetRasterBand(1).GetMaskBand().ReadAsArray() > 0
            mask_b = original.GetRasterBand(1).GetMaskBand().ReadAsArray() > 0
            if not np.array_equal(mask_a, mask_b):
                raise ValueError("Recomputed precipitation mask differs")
            maximum_difference = float(np.max(np.abs(a[mask_a] - b[mask_a])))
            if maximum_difference > 0.001:
                raise ValueError("Native precipitation accumulation does not reproduce M2")
        for path in [*rasters.values(), partition / "zones.tif"]:
            verified_artifact(path)
        stats = cache.build(
            "statistics",
            [
                *rasters.values(),
                partition / "zones.tif",
                partition / "zone_station.csv",
                index_path,
                *implementation,
            ],
            {"year": year},
            lambda out, files=rasters, y=year: station_statistics(index, files, y, out),
        )
        with (stats / "stations.csv").open(encoding="utf-8-sig") as stream:
            records.extend(csv.DictReader(stream))
        artifacts.append(
            {
                "year": year,
                "native": str(native),
                "statistics": str(stats),
                "native_precip_max_abs_difference_mm": maximum_difference,
            }
        )
    output = root / "project/m4/climate_review"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "stations.csv", records)
    write_json(
        output / "summary.json",
        {
            "status": "diagnostic_complete_not_calibrated",
            "artifacts": artifacts,
            "elapsed_seconds": time.perf_counter() - started,
            "validation_used": False,
            "production_inputs_changed": False,
            "region_boundaries": "unavailable_user_confirmed_proxy_only",
            "cache_events": cache.events,
        },
    )


if __name__ == "__main__":
    main()
