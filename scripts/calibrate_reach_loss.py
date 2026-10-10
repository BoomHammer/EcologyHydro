"""Compare one unresolved reach-loss term with the frozen four-Z diagnostics."""

import argparse
import multiprocessing
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import psutil
from calibrate_regional_water import (
    YEARS,
    full_year,
    metrics,
    sample_year,
    sampling_positions,
    source_paths,
    water_accounts,
)
from osgeo import gdal

from ecologyhydro.aggregation import write_csv
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.config import load_config, project_root
from ecologyhydro.reach_loss import apply_reach_loss, fit_reach_loss
from ecologyhydro.regional_calibration import ENDPOINTS, predict, route
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.water_balance import read_csv


def worker(product, output):
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    root, output = project_root(), Path(output)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    tables = read_json(root / "project/calibration/regional_water_v1/base_tables.json")[product]
    paths = {y: source_paths(root, index, product, y) for y in YEARS}
    positions, weights, volume = sampling_positions(paths[2019])
    samples = {y: sample_year(paths[y], tables[str(y)], positions, weights) for y in YEARS[:4]}
    data = {k: np.concatenate([samples[y][k] for y in YEARS[:4]]) for k in samples[2019]}
    data["groups"] = np.concatenate([samples[y]["zone"] + i * 11 for i, y in enumerate(YEARS[:4])])
    obs = {r["测站"]: r for r in read_csv(root / "data/Hydrology/实测年径流量2018-2023.csv")}
    stations = load_config().study.station_ids
    target = np.array([[float(obs[stations[s]][str(y)]) for s in ENDPOINTS] for y in YEARS])
    cases, cv_rows = {}, []
    for version in ("legacy", "sector_2022"):
        consumption, storage = water_accounts(root, version)
        adjustment = (consumption + storage).cumsum(axis=1)
        fitted = fit_reach_loss(data, target[:4], adjustment[:4])
        predicted, fold_parameters = [], []
        for withheld in range(4):
            fold = fit_reach_loss(data, target[:4], adjustment[:4], keep=np.arange(4) != withheld)
            estimate = apply_reach_loss(
                predict(fold["parameters"], data)[:, ENDPOINTS] - adjustment[:4],
                fold["annual_loss_1e8_m3"],
            )[withheld]
            predicted.append(estimate)
            fold_parameters.append({"withheld_year": YEARS[withheld], **fold})
            for i, station in enumerate(ENDPOINTS):
                cv_rows.append(
                    {
                        "landcover": product,
                        "variant": version,
                        "year": YEARS[withheld],
                        "station": stations[station],
                        "predicted": estimate[i],
                        "observed": target[withheld, i],
                        "annual_loss_1e8_m3": fold["annual_loss_1e8_m3"],
                    }
                )
        cases[version] = {
            **fitted,
            "development_leave_year_out": metrics(np.array(predicted), target[:4]),
            "folds": fold_parameters,
        }
        print(product, version, fitted, flush=True)
    write_json(output / f"parameters_{product}.json", cases)
    write_csv(output / f"cv_stations_{product}.csv", cv_rows)
    samples[2023] = sample_year(paths[2023], tables["2023"], positions, weights)
    complete = {
        y: full_year(
            paths[y], tables[str(y)], {k: c["parameters"] for k, c in cases.items()}, volume
        )
        for y in YEARS
    }
    rows, scores = [], {}
    for version, case in cases.items():
        consumption, storage = water_accounts(root, version)
        natural = np.array([complete[y][version][ENDPOINTS] for y in YEARS])
        base = route(natural, consumption, storage)
        estimates = apply_reach_loss(base, case["annual_loss_1e8_m3"])
        approximate = np.vstack(
            [
                predict(case["parameters"], data)[:, ENDPOINTS],
                predict(case["parameters"], samples[2023], 1)[:, ENDPOINTS],
            ]
        )
        discrepancy = float(np.max(np.abs(approximate - natural) / target))
        if discrepancy > 0.01:
            raise ValueError("Spatial quadrature discrepancy exceeds 1% of observation")
        scores[version] = {
            "training": metrics(estimates[:4], target[:4]),
            "development_leave_year_out": case["development_leave_year_out"],
            "reused_2023": metrics(estimates[4], target[4]),
            "quadrature_max_fraction_observed": discrepancy,
            "annual_loss_1e8_m3": case["annual_loss_1e8_m3"],
        }
        for yi, year in enumerate(YEARS):
            for si, station in enumerate(ENDPOINTS):
                rows.append(
                    {
                        "landcover": product,
                        "variant": version,
                        "year": year,
                        "station": stations[station],
                        "awy_yield": natural[yi, si],
                        "consumption_cumulative": consumption[yi, : si + 1].sum(),
                        "storage_cumulative": storage[yi, : si + 1].sum(),
                        "net_loss_cumulative": case["annual_loss_1e8_m3"] if si >= 2 else 0,
                        "predicted": estimates[yi, si],
                        "observed": target[yi, si],
                        "relative_error": estimates[yi, si] / target[yi, si] - 1,
                    }
                )
    write_csv(output / f"stations_{product}.csv", rows)
    write_json(output / f"summary_{product}.json", scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/reach_loss_v1")
    args = parser.parse_args()
    root = project_root()
    configure_threads(load_config().resources)
    output = (root / args.output).resolve()
    if not output.is_relative_to(root.resolve() / "project/calibration"):
        raise ValueError("Output must be inside project/calibration")
    output.mkdir(exist_ok=False)
    write_json(
        output / "protocol.json",
        {
            "training_years": YEARS[:4],
            "reused_test_year": 2023,
            "hypothesis": "Known LZ--TDG accounts imply negative yield in all development years",
            "parameters": "four Zs and common vegetated Kc multiplier plus one constant reach loss",
            "bounds": {"z": [1, 30], "kc_multiplier": [0.7, 2], "annual_loss_1e8_m3": [0, 100]},
            "loss_location": "Lanzhou--Toudaoguai; subtract once from TDG and downstream",
            "interpretation": "Unidentified net-account discrepancy, NOT measured seepage",
            "versions": ["legacy", "sector_2022"],
            "comparator": "regional_water_v2, five parameters versus six here",
            "objective": "equal station-year relative squared error, three predefined starts",
            "evaluation": "refit all six parameters for each leave-year-out fold; both products",
            "climate_selection": "previously selected using all development years, unchanged",
            "independent_validation": False,
            "future_use": "Not approved for CMIP6 or land-cover causal scenarios",
            "mask_policy": "No inland/desert cells removed; spatial audit is a separate hypothesis",
            "quadrature_tolerance": "max sample/full difference / observed runoff <= 1%",
        },
    )
    sources = [
        Path(__file__),
        root / "src/ecologyhydro/reach_loss.py",
        root / "src/ecologyhydro/regional_calibration.py",
        root / "scripts/calibrate_regional_water.py",
        root / "project/calibration/regional_water_v1/manifest.json",
        root / "project/calibration/regional_water_v1/base_tables.json",
        root / "project/calibration/regional_water_v2/summary.json",
        output / "protocol.json",
    ]
    write_json(output / "manifest.json", {"sources": [fingerprint(p) for p in sources]})
    started, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(p, str(output))) for p in ("fine", "copernicus")
    ]
    for process in processes:
        process.start()
    try:
        while any(p.is_alive() for p in processes):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if memory > 24 * 1024**3 or time.perf_counter() - started > 12 * 3600:
                raise RuntimeError("Resource budget exceeded")
            if any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError("Worker failed")
            time.sleep(1)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError("Worker failed")
    write_json(
        output / "summary.json",
        {
            "products": {
                p: read_json(output / f"summary_{p}.json") for p in ("fine", "copernicus")
            },
            "elapsed_seconds_workers": time.perf_counter() - started,
            "peak_process_tree_mib": peak / 1024**2,
            "workers": 2,
        },
    )
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
