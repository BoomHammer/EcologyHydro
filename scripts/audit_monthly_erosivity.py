"""Audit local CMFD and prepare compact monthly-rainfall proxies, not validated R."""

import argparse
import calendar
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

from ecologyhydro.cache import fingerprint
from ecologyhydro.climate import axis_window
from ecologyhydro.simulation import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", choices=("prec", "bcpr"), default="prec")
    args = parser.parse_args()
    source = next(Path("data/Climate/Data_forcing_01mo_010deg").glob(f"{args.variable}_*.nc"))
    grid_path = Path("project/repairs/soil_routing/m2_index.json")
    bounds = json.loads(grid_path.read_text(encoding="utf-8"))["geographic_bounds"]
    output = Path("project/sediment/monthly_climate_audit") / args.variable
    output.mkdir(parents=True, exist_ok=True)
    years = np.arange(1991, 2025)
    precipitation, mfi = [], []
    with Dataset(source) as dataset:
        variable = dataset[args.variable]
        if variable.dimensions != ("time", "lat", "lon") or variable.units != "kg m-2 s-1":
            raise ValueError("Unexpected CMFD precipitation layout/units")
        time = dataset["time"]
        cal = getattr(time, "calendar", "standard")
        if cal not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError(f"Unsupported calendar: {cal}")
        dates = num2date(time[:], time.units, calendar=cal)
        xs = axis_window(dataset["lon"][:], bounds[0], bounds[2])
        ys = axis_window(dataset["lat"][:], bounds[1], bounds[3])
        lon, lat = dataset["lon"][xs], dataset["lat"][ys]
        shape = (len(lat), len(lon))
        for year in years:
            months = [(i, date.month) for i, date in enumerate(dates) if date.year == year]
            if sorted(m for _, m in months) != list(range(1, 13)):
                raise ValueError(f"Missing or duplicate months: {year}")
            total = np.zeros(shape, dtype=np.float64)
            squares = total.copy()
            valid = np.ones(shape, dtype=bool)
            for index, month in months:
                flux = np.ma.asarray(variable[index, ys, xs], dtype=np.float64)
                depth = flux.filled(np.nan) * calendar.monthrange(int(year), month)[1] * 86400
                valid &= np.isfinite(depth) & (depth >= 0)
                total += depth
                squares += depth**2
            proxy = np.divide(squares, total, out=np.zeros_like(total), where=total > 0)
            if np.any(valid & ((proxy < total / 12 - 1e-7) | (proxy > total + 1e-7))):
                raise ValueError("MFI violates mathematical bounds")
            precipitation.append(np.where(valid, total, np.nan))
            mfi.append(np.where(valid, proxy, np.nan))
    precipitation, mfi = np.asarray(precipitation), np.asarray(mfi)
    baseline = mfi[years <= 2020].mean(axis=0)
    valid = np.isfinite(mfi).all(axis=0) & (baseline > 0)
    ratio = np.divide(mfi, baseline, out=np.full_like(mfi, np.nan), where=valid)
    if not valid.any():
        raise ValueError("No complete baseline climate cells")
    error = float(np.max(np.abs(ratio[years <= 2020, :, :][:, valid].mean(axis=0) - 1)))
    if error > 1e-10:
        raise ValueError("Baseline anomaly does not average to one")
    artifact = output / "annual_monthly_proxies.npz"
    np.savez_compressed(
        artifact,
        years=years,
        longitude=lon,
        latitude=lat,
        annual_precipitation_mm=precipitation.astype(np.float32),
        mfi_mm=mfi.astype(np.float32),
        mfi_anomaly=ratio.astype(np.float32),
        complete_baseline_mask=valid,
    )
    record = {
        "status": "monthly_input_available_proxy_only_not_validated_erosivity",
        "source": fingerprint(source),
        "grid_source": fingerprint(grid_path),
        "code": fingerprint(Path(__file__)),
        "artifact": fingerprint(artifact),
        "period": [1991, 2024],
        "baseline": [1991, 2020],
        "months_read": 408,
        "subset_shape": shape,
        "complete_native_cells": int(valid.sum()),
        "total_native_cells": int(valid.size),
        "subset_months_float32_mib": 408 * valid.size * 4 / 1024**2,
        "artifact_mib": artifact.stat().st_size / 1024**2,
        "baseline_mean_anomaly_max_error": error,
        "formula": "MFI_y = sum(P_month_mm**2) / sum(P_month_mm)",
        "zero_annual_precipitation": "MFI=0; zero baseline remains invalid for ratios",
        "interpretation": "Monthly rainfall concentration proxy; MFI is not R or EI30. "
        "Baseline-normalized MFI is an unvalidated temporal-disaggregation hypothesis.",
        "domain": "Bounding rectangle plus interpolation buffer; not a basin-mean statistic",
        "new_download_bytes": 0,
        "station_observations_used": False,
        "references": [
            "https://doi.org/10.1155/2013/256352",
            "https://doi.org/10.5194/hess-19-4113-2015",
        ],
    }
    write_json(output / "audit.json", record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "months_read",
                    "subset_shape",
                    "complete_native_cells",
                    "artifact_mib",
                    "baseline_mean_anomaly_max_error",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
