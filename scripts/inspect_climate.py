"""Inspect NetCDF headers and coordinate axes without reading climate arrays."""

import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

from ecologyhydro.config import load_config


def attributes(variable):
    return {key: variable.getncattr(key) for key in variable.ncattrs()}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)}")


def inspect(path: Path) -> dict:
    with Dataset(path, "r") as dataset:
        time = dataset.variables["time"]
        dates = num2date(time[:], time.units, calendar=getattr(time, "calendar", "standard"))
        selected = [date for date in dates if 2019 <= date.year <= 2023]
        months = [(date.year, date.month) for date in selected]
        expected = {(year, month) for year in range(2019, 2024) for month in range(1, 13)}
        axes = {}
        for name in ("lon", "lat"):
            values = np.asarray(dataset.variables[name][:])
            differences = np.diff(values)
            axes[name] = {
                "size": values.size,
                "first": values[0],
                "last": values[-1],
                "minimum": values.min(),
                "maximum": values.max(),
                "step_min": differences.min(),
                "step_max": differences.max(),
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            }
        return {
            "file": path.name,
            "format": dataset.data_model,
            "global_attributes": attributes(dataset),
            "dimensions": {name: len(value) for name, value in dataset.dimensions.items()},
            "variables": {
                name: {
                    "dimensions": value.dimensions,
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "attributes": attributes(value),
                }
                for name, value in dataset.variables.items()
            },
            "axes": axes,
            "time": {
                "first": str(dates[0]),
                "last": str(dates[-1]),
                "selected_dates": [str(date) for date in selected],
                "complete_2019_2023": set(months) == expected and len(months) == 60,
                "strictly_increasing": bool(np.all(np.diff(time[:]) > 0)),
            },
        }


def main():
    config = load_config()
    files = sorted(config.paths.climate.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files under {config.paths.climate}")
    records = [inspect(path) for path in files]
    same_axes = all(record["axes"] == records[0]["axes"] for record in records)
    same_dates = all(record["time"] == records[0]["time"] for record in records)
    report = {
        "scope": "Headers and coordinate axes only; climate values not read",
        "matching_spatial_axes": same_axes,
        "matching_time_axes": same_dates,
        "files": records,
    }
    output = config.paths.cache / "inventory" / "climate_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_value), encoding="utf-8"
    )
    print(f"Report: {output}")
    print(f"Matching spatial/time axes: {same_axes}/{same_dates}")
    for record in records:
        name = record["file"].split("_")[0]
        print(name, record["variables"][name], record["time"]["complete_2019_2023"])
    print("Coordinates:", json.dumps(records[0]["axes"], default=json_value))
    print("Time:", records[0]["time"])


if __name__ == "__main__":
    main()
