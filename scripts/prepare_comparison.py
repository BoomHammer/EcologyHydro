"""Prepare complete YAML-selected P/PET years and the third landcover, without refitting."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from osgeo import gdal
from prepare_long_climate import monthly_data

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import accounts, load_protocol, resolve
from ecologyhydro.lakes import overlay_lakes
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import NODATA, warp, write_raster


def raw_source(config, variable, year):
    climate = config["climate"]
    for entry in climate["reuse"]:
        if year in entry["years"]:
            path = resolve(config, entry["raw_template"], variable=variable, year=year)
            if path.is_file():
                return path
    return resolve(config, climate["raw_template"], variable=variable, year=year)


def fetch(config, variable, year):
    path = raw_source(config, variable, year)
    if path.is_file():
        record = read_json(path.with_suffix(".json"))
        if fingerprint(path)["sha256"] != record["file"]["sha256"]:
            raise ValueError(f"Raw climate fingerprint mismatch: {path}")
        return variable, year, path
    path.parent.mkdir(parents=True, exist_ok=True)
    climate = config["climate"]
    query = urlencode(
        {
            "var": variable,
            **climate["bounds"],
            "horizStride": 1,
            "time_start": f"{year}-01-01T00:00:00Z",
            "time_end": f"{year}-12-31T23:59:59Z",
            "timeStride": 1,
            "accept": "netcdf",
        }
    )
    url = climate["url"].format(variable=variable) + "?" + query
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


def prepare(config):
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    accounts(config)  # Check labels and required year columns before network requests.
    output = resolve(config, config["paths"]["prepared"])
    output.mkdir(parents=True, exist_ok=True)
    grid_path = resolve(config, config["paths"]["grid_index"])
    grid = read_json(grid_path)["grid"]
    previous = read_json(output / "prepared.json") if (output / "prepared.json").is_file() else {}
    reusable_grid = previous.get("grid", {}).get("sha256") == fingerprint(grid_path)["sha256"]
    old_records = {(r["variable"], r["year"]): r for r in previous.get("climate", [])}
    jobs = [(v, y) for y in config["forcing_years"] for v in config["climate"]["variables"]]
    if config["climate"]["version"] != "V1.1" or config["climate"]["variables"] != ["ppt", "pet"]:
        raise ValueError("This protocol requires V1.1 precipitation and PET")
    with ThreadPoolExecutor(max_workers=config["resources"]["max_workers"]) as pool:
        downloads = list(pool.map(lambda job: fetch(config, *job), jobs))
    records = []
    # netCDF C decoding and GDAL writes are sequential; network requests overlap only.
    for variable, year, path in downloads:
        values, gt, metadata = monthly_data(path, variable, year)
        aligned = resolve(
            config, config["climate"]["aligned_template"], variable=variable, year=year
        )
        aligned.parent.mkdir(parents=True, exist_ok=True)
        prior = old_records.get((variable, year))
        if (
            reusable_grid
            and prior
            and prior["raw"]["sha256"] == fingerprint(path)["sha256"]
            and aligned.is_file()
            and prior["aligned"]["sha256"] == fingerprint(aligned)["sha256"]
        ):
            records.append({**prior, "raw": fingerprint(path), "aligned": fingerprint(aligned)})
            print("Verified and reused", variable, year, flush=True)
            continue
        native = aligned.with_name(aligned.stem + "_native.tif")
        total = values.sum(axis=0)
        write_raster(native, np.where(np.isfinite(total), total, NODATA), gt, "EPSG:4326")
        warp(native, aligned, grid)
        records.append(
            {
                "variable": variable,
                "year": year,
                "months": 12,
                "raw": fingerprint(path),
                "aligned": fingerprint(aligned),
                "metadata": metadata,
            }
        )
        print("Verified 12 months and aligned", variable, year, flush=True)
    overlays = {}
    for product, settings in config["products"].items():
        if "native_aligned" not in settings:
            continue
        aligned = resolve(config, settings["aligned"])
        aligned.parent.mkdir(parents=True, exist_ok=True)
        native = resolve(config, settings["native_aligned"])
        warp(resolve(config, settings["source"]), native, grid, method="near")
        overlays[product] = overlay_lakes(
            native,
            resolve(config, config["paths"]["lake_mask"]),
            settings["water_code"],
            aligned.parent,
        )
    write_json(
        output / "prepared.json",
        {
            "config": fingerprint(config["config_path"]),
            "grid": fingerprint(grid_path),
            "climate": records,
            "lake_overlays": overlays,
            "landcover": {
                p: fingerprint(resolve(config, s["aligned"])) for p, s in config["products"].items()
            },
        },
    )
    print(output / "prepared.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare(load_protocol(args.config))


if __name__ == "__main__":
    main()
