"""Fit three products on training years only, then evaluate all nine fixed-parameter pairs."""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import yaml
from calibrate_long_record import full_models
from calibrate_regional_water import sample_year, sampling_positions
from osgeo import gdal, osr
from prepare_comparison import raw_source
from prepare_long_climate import monthly_data
from scipy.ndimage import map_coordinates

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.comparison import (
    accounts,
    load_protocol,
    phase_for,
    prediction_rows,
    resolve,
    summarize_rows,
    training_mask,
)
from ecologyhydro.config import Resources
from ecologyhydro.long_record import ENDPOINTS
from ecologyhydro.regional_transfer import evaluate, fit_transfer
from ecologyhydro.rotation import weighted_kc
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.spatial import crs


def input_paths(config, product, year):
    paths = config["paths"]
    return {
        "zone": str(resolve(config, paths["zones"])),
        "code": str(resolve(config, config["products"][product]["aligned"])),
        "pawc": str(resolve(config, paths["pawc"])),
        "soil": str(resolve(config, paths["soil"])),
        "p": str(resolve(config, config["climate"]["aligned_template"], variable="ppt", year=year)),
        "et": str(
            resolve(config, config["climate"]["aligned_template"], variable="pet", year=year)
        ),
    }


def product_tables(config, product, positions, weights):
    settings = config["products"][product]
    base = read_json(resolve(config, config["paths"]["base_tables"]))
    original = deepcopy(base[settings["table_product"]][settings["table_year"]])
    if "code_transfer" in settings:
        lookup = {int(row["lucode"]): row for row in original}
        original = [
            {**deepcopy(lookup[source]), "lucode": code}
            for code, source in settings["code_transfer"].items()
        ]
    tables = {year: deepcopy(original) for year in config["forcing_years"]}
    rotation = settings["rotation_codes"]
    if not rotation:
        return tables, []
    with gdal.Open(str(resolve(config, settings["aligned"]))) as raster:
        selected = np.isin(raster.ReadAsArray().ravel()[positions], rotation)
        if not selected.any():
            raise ValueError("No configured crop rotation pixels in sampling strata")
        yy, xx = np.divmod(positions[selected], raster.RasterXSize)
        gt = raster.GetGeoTransform()
        xy = np.column_stack(
            [
                gt[0] + (xx + 0.5) * gt[1] + (yy + 0.5) * gt[2],
                gt[3] + (xx + 0.5) * gt[4] + (yy + 0.5) * gt[5],
            ]
        )
        transform = osr.CoordinateTransformation(crs(raster.GetProjection()), crs("EPSG:4326"))
        lonlat = np.array(transform.TransformPoints(xy))
    crop = yaml.safe_load(
        resolve(config, config["paths"]["crop_systems"]).read_text(encoding="utf-8")
    )
    coefficients = crop["systems"]["winter_wheat_summer_maize"]["monthly_kc"]
    records = []
    for year in config["forcing_years"]:
        path = raw_source(config, "pet", year)
        monthly, gt, _ = monthly_data(path, "pet", year)
        coordinates = [(lonlat[:, 1] - gt[3]) / gt[5] - 0.5, (lonlat[:, 0] - gt[0]) / gt[1] - 0.5]
        sums = []
        for values in monthly:
            values = map_coordinates(values, coordinates, order=1, mode="constant", cval=np.nan)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Missing crop PET")
            sums.append(float(values @ weights[selected]))
        kc = weighted_kc(sums, coefficients)
        for row in tables[year]:
            if int(row["lucode"]) in rotation:
                row["kc"] = kc
        records.append({"year": year, "kc": kc, "pet_source": str(path)})
    return tables, records


def worker_setup(config_path):
    config = load_protocol(config_path)
    configure_threads(Resources.model_validate(config["resources"]))
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    return config


def fit_product(config_path, output_string, product):
    config, output = worker_setup(config_path), Path(output_string)
    years = config["forcing_years"]
    paths = {y: input_paths(config, product, y) for y in years}
    positions, weights, _ = sampling_positions(
        paths[years[0]], config["evaluation"]["sampling_budget_per_zone"]
    )
    tables, rotation = product_tables(config, product, positions, weights)
    write_json(output / f"tables_{product}.json", tables)
    if rotation:
        write_csv(output / f"rotation_{product}.csv", rotation)
    samples = {}
    for year in years:
        samples[year] = sample_year(paths[year], tables[year], positions, weights)
        print("Sampled", product, year, flush=True)
    data = {key: np.concatenate([samples[y][key] for y in years]) for key in samples[years[0]]}
    data["groups"] = np.concatenate([samples[y]["zone"] + i * 11 for i, y in enumerate(years)])
    np.savez_compressed(output / f"samples_{product}.npz", **data)
    target, adjustment, eligible, _, _ = accounts(config)
    fits = {}
    for name, stage1 in (("stage1_diagnostic", True), ("final", False)):
        fits[name] = fit_transfer(
            data, target, adjustment, eligible, training_mask(config, stage1), config["model"]
        )
        fits[name]["training_years"] = [years[i] for i in fits[name]["training_indices"]]
        print("Fitted", product, name, fits[name]["parameters"], flush=True)
    write_json(output / f"parameters_{product}.json", fits)
    return product


def evaluate_product(config_path, output_string, product):
    config, output = worker_setup(config_path), Path(output_string)
    years = config["forcing_years"]
    tables = read_json(output / f"tables_{product}.json")
    fits = {p: read_json(output / f"parameters_{p}.json")["final"] for p in config["products"]}
    cases = {p: (config["model"], fit) for p, fit in fits.items()}
    with np.load(output / f"samples_{product}.npz") as sample_file:
        data = {key: sample_file[key] for key in sample_file.files}
    approximate = {
        p: evaluate(f["parameters"], data, config["model"], len(years)) for p, f in fits.items()
    }
    del data
    target, adjustment, eligible, _, _ = accounts(config)
    natural = {p: [] for p in fits}
    raw_rows = []
    paths = input_paths(config, product, years[0])
    with gdal.Open(paths["zone"]) as raster:
        gt = raster.GetGeoTransform()
        volume = abs(gt[1] * gt[5] - gt[2] * gt[4]) / 1e11
    for year in years:
        official, _ = full_models(
            input_paths(config, product, year), tables[str(year)], cases, volume, all_stations=True
        )
        for source, values in official.items():
            natural[source].append(values[ENDPOINTS])
            for si, station in enumerate(config["evaluation"]["all_stations"]):
                raw_rows.append(
                    {
                        "parameter_source": source,
                        "landcover": product,
                        "year": year,
                        "phase": phase_for(config, year),
                        "station": station,
                        "awy_yield": float(values[si]),
                        "meaning": "cumulative_AWY_yield_not_managed_station_runoff",
                    }
                )
        print("Official full-pixel 3 parameter sets", product, year, flush=True)
    rows, summaries, checks = [], [], {}
    for source, fit in fits.items():
        produced = np.array(natural[source])
        before_account = produced - np.array([0, 0, 1, 1, 1, 1, 1]) * fit["parameters"][-1]
        predicted = before_account - adjustment
        error = float(
            np.max(
                np.abs(before_account[eligible] - approximate[source][eligible]) / target[eligible]
            )
        )
        checks[source] = error
        pair_rows = prediction_rows(
            config, source, product, predicted, produced, target, adjustment, eligible
        )
        rows.extend(pair_rows)
        summaries.extend(summarize_rows(pair_rows))
    write_csv(output / f"predictions_{product}.csv", rows)
    write_csv(output / f"metrics_{product}.csv", summaries)
    write_csv(output / f"awy_all_stations_{product}.csv", raw_rows)
    write_json(output / f"quadrature_{product}.json", checks)
    if max(checks.values()) > config["evaluation"]["quadrature_max_fraction_observed"]:
        raise ValueError(f"{product}: sampling error exceeded configured tolerance: {checks}")
    return product


def run_pool(function, config, output):
    started = time.perf_counter()
    peak = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config["resources"]["max_workers"], mp_context=context
    ) as pool:
        futures = [
            pool.submit(function, str(config["config_path"]), str(output), p)
            for p in config["products"]
        ]
        while not all(f.done() for f in futures):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if (
                memory > config["resources"]["memory_budget_gb"] * 1024**3
                or time.perf_counter() - started > config["resources"]["timeout_hours"] * 3600
            ):
                for child in psutil.Process().children(recursive=True):
                    with suppress(psutil.NoSuchProcess):
                        child.terminate()
                raise RuntimeError("Configured process memory/time budget exceeded")
            for future in futures:
                if future.done():
                    future.result()
            time.sleep(1)
        for future in futures:
            future.result()
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_tree_mib": peak / 1024**2,
    }


def validate_prepared(config):
    record = read_json(resolve(config, config["paths"]["prepared"]) / "prepared.json")
    if record["config"]["sha256"] != fingerprint(config["config_path"])["sha256"]:
        raise ValueError("Configuration changed; run prepare_comparison again")
    expected = {(v, y) for v in config["climate"]["variables"] for y in config["forcing_years"]}
    if {(r["variable"], r["year"]) for r in record["climate"]} != expected:
        raise ValueError("Incomplete prepared climate years")
    for row in record["climate"]:
        for key in ("raw", "aligned"):
            artifact = row[key]
            if fingerprint(Path(artifact["path"]))["sha256"] != artifact["sha256"]:
                raise ValueError("Prepared climate fingerprint changed")
    for artifact in record["landcover"].values():
        if fingerprint(Path(artifact["path"]))["sha256"] != artifact["sha256"]:
            raise ValueError("Prepared landcover fingerprint changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_protocol(args.config)
    validate_prepared(config)
    target, adjustment, eligible, consumption, storage = accounts(config)
    output_root = resolve(config, config["paths"]["output_root"])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    protocol = {k: v for k, v in config.items() if k not in ("root", "config_path")}
    protocol.update(
        {
            "fit_policy": "union_of_stage1_and_stage2_no_validation_refit",
            "unit": "1e8 m3/year",
            "warmup_note": "AWY is stateless; computed but unscored, no state initialization claim",
            "validation_caveat": (
                "2022 and 2023 were examined by historical model development; 2024 newly added"
            ),
            "account_note": (
                "No interpolation or zero filling; missing upstream account propagates downstream"
            ),
        }
    )
    write_json(output / "protocol.json", protocol)
    sources = [config["config_path"], Path(__file__)]
    sources.extend(resolve(config, p) for p in config["hydrology"].values())
    sources.extend(
        resolve(config, p)
        for k, p in config["paths"].items()
        if k not in ("prepared", "output_root")
    )
    sources.extend(
        resolve(config, s[k])
        for s in config["products"].values()
        for k in ("source", "classes", "aligned")
    )
    write_json(output / "manifest.json", {"sources": [fingerprint(p) for p in sources]})
    write_csv(
        output / "accounts.csv",
        [
            {
                "year": y,
                "station": s,
                "observed": None if not np.isfinite(target[yi, si]) else float(target[yi, si]),
                "consumption": None
                if not np.isfinite(consumption[yi, si])
                else float(consumption[yi, si]),
                "storage": None if not np.isfinite(storage[yi, si]) else float(storage[yi, si]),
                "eligible_account": bool(eligible[yi, si]),
            }
            for yi, y in enumerate(config["forcing_years"])
            for si, s in enumerate(config["evaluation"]["managed_stations"])
        ],
    )
    print("Run directory:", output, flush=True)
    performance = {"fitting": run_pool(fit_product, config, output)}
    # All three parameter files are committed before any validation forward calculation.
    performance["evaluation"] = run_pool(evaluate_product, config, output)
    from ecologyhydro.water_balance import read_csv

    for stem in ("predictions", "metrics", "awy_all_stations"):
        rows = [row for p in config["products"] for row in read_csv(output / f"{stem}_{p}.csv")]
        write_csv(output / f"{stem}.csv", rows)
    write_json(
        output / "summary.json",
        {
            "status": "complete",
            "pairs": 9,
            "years": config["forcing_years"],
            "training_years": np.array(config["forcing_years"])[training_mask(config)].tolist(),
            "validation_years": config["years"]["validation"],
            "performance": performance,
            "managed_station_count": 7,
            "raw_yield_station_count": 11,
        },
    )
    print("Completed:", output, flush=True)


if __name__ == "__main__":
    main()
