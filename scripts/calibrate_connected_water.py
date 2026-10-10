"""Refit fixed inland-domain and annual soil/snow-account hypotheses fairly."""

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
from ecologyhydro.regional_calibration import ENDPOINTS, fit, predict
from ecologyhydro.runtime import configure_threads
from ecologyhydro.simulation import write_json
from ecologyhydro.storage_diagnostic import cumulative_storage
from ecologyhydro.water_balance import read_csv


def estimate_fit(mode, data, target, adjustment, keep=None):
    if mode == "domain_only":
        return {
            **fit(data, target, adjustment, regional=True, regions=4, keep=keep),
            "annual_loss_1e8_m3": 0.0,
        }
    return fit_reach_loss(data, target, adjustment, keep=keep)


def worker(product, output, rainfall=False):
    gdal.UseExceptions()
    gdal.SetCacheMax(128 * 1024**2)
    root, output = project_root(), Path(output)
    index = read_json(root / "project/repairs/soil_routing/m2_index.json")
    tables = read_json(root / "project/calibration/regional_water_v1/base_tables.json")[product]
    paths = {
        year: {
            **source_paths(root, index, product, year),
            "zone": str(root / "project/diagnostics/gonghe_connectivity_v1/zones.tif"),
        }
        for year in YEARS
    }
    if rainfall:
        for year in YEARS:
            paths[year]["p"] = str(
                root / f"project/diagnostics/reference_precipitation_v1/precipitation_{year}.tif"
            )
    positions, weights, volume = sampling_positions(paths[2019])
    samples = {
        year: sample_year(paths[year], tables[str(year)], positions, weights) for year in YEARS[:4]
    }
    data = {
        key: np.concatenate([samples[year][key] for year in YEARS[:4]]) for key in samples[2019]
    }
    data["groups"] = np.concatenate(
        [samples[year]["zone"] + yi * 11 for yi, year in enumerate(YEARS[:4])]
    )
    obs = {r["测站"]: r for r in read_csv(root / "data/Hydrology/实测年径流量2018-2023.csv")}
    stations = load_config().study.station_ids
    target = np.array([[float(obs[stations[s]][str(year)]) for s in ENDPOINTS] for year in YEARS])
    storage_proxy = cumulative_storage(
        read_csv(root / "project/diagnostics/storage_change_v1/storage_changes.csv"),
        "gonghe_excluded",
    )
    cases, adjustments, cv_rows = {}, {}, []
    modes = ("domain_loss",) if rainfall else ("domain_only", "domain_loss", "domain_storage_loss")
    for mode in modes:
        for version in ("legacy", "sector_2022"):
            key = f"{mode}__{version}"
            consumption, reservoir = water_accounts(root, version)
            adjustment = (consumption + reservoir).cumsum(axis=1)
            if mode == "domain_storage_loss":
                adjustment += storage_proxy
            adjustments[key] = adjustment
            fitted = estimate_fit(mode, data, target[:4], adjustment[:4])
            predictions, folds = [], []
            for withheld in range(4):
                fold = estimate_fit(
                    mode, data, target[:4], adjustment[:4], np.arange(4) != withheld
                )
                estimate = apply_reach_loss(
                    predict(fold["parameters"], data)[:, ENDPOINTS] - adjustment[:4],
                    fold["annual_loss_1e8_m3"],
                )[withheld]
                predictions.append(estimate)
                folds.append({"withheld_year": YEARS[withheld], **fold})
                for si, station in enumerate(ENDPOINTS):
                    cv_rows.append(
                        {
                            "landcover": product,
                            "variant": key,
                            "year": YEARS[withheld],
                            "station": stations[station],
                            "predicted": estimate[si],
                            "observed": target[withheld, si],
                        }
                    )
            cases[key] = {
                **fitted,
                "folds": folds,
                "development_leave_year_out": metrics(np.array(predictions), target[:4]),
            }
            print(product, key, "LOYO", cases[key]["development_leave_year_out"], flush=True)
    write_json(output / f"parameters_{product}.json", cases)
    write_csv(output / f"cv_stations_{product}.csv", cv_rows)
    samples[2023] = sample_year(paths[2023], tables["2023"], positions, weights)
    complete = {
        year: full_year(
            paths[year],
            tables[str(year)],
            {key: case["parameters"] for key, case in cases.items()},
            volume,
        )
        for year in YEARS
    }
    rows, scores = [], {}
    for key, case in cases.items():
        natural = np.array([complete[year][key][ENDPOINTS] for year in YEARS])
        estimate = apply_reach_loss(natural - adjustments[key], case["annual_loss_1e8_m3"])
        approximate = np.vstack(
            [
                predict(case["parameters"], data)[:, ENDPOINTS],
                predict(case["parameters"], samples[2023], 1)[:, ENDPOINTS],
            ]
        )
        discrepancy = float(np.max(np.abs(approximate - natural) / target))
        if discrepancy > 0.01:
            raise ValueError("Quadrature discrepancy exceeds 1% of observations")
        scores[key] = {
            "training": metrics(estimate[:4], target[:4]),
            "development_leave_year_out": case["development_leave_year_out"],
            "reused_2023": metrics(estimate[4], target[4]),
            "quadrature_max_fraction_observed": discrepancy,
            "annual_loss_1e8_m3": case["annual_loss_1e8_m3"],
        }
        for yi, year in enumerate(YEARS):
            for si, station in enumerate(ENDPOINTS):
                rows.append(
                    {
                        "landcover": product,
                        "variant": key,
                        "year": year,
                        "station": stations[station],
                        "awy_yield": natural[yi, si],
                        "known_adjustment_cumulative": adjustments[key][yi, si],
                        "soil_snow_delta_cumulative": storage_proxy[yi, si]
                        if key.startswith("domain_storage")
                        else 0,
                        "net_loss_cumulative": case["annual_loss_1e8_m3"] if si >= 2 else 0,
                        "predicted": estimate[yi, si],
                        "observed": target[yi, si],
                        "relative_error": estimate[yi, si] / target[yi, si] - 1,
                    }
                )
    write_csv(output / f"stations_{product}.csv", rows)
    write_json(output / f"summary_{product}.json", scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/calibration/connected_water_v1")
    parser.add_argument("--reference-rainfall", action="store_true")
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
            "independent_validation": False,
            "domain": "HydroBASINS inland polygon 4060051460, Chaka/Shazhuyu part of Gonghe basin",
            "boundary_fitted_to_runoff_or_area": False,
            "models": ["domain_loss"]
            if args.reference_rainfall
            else ["domain_only", "domain_loss", "domain_storage_loss"],
            "parameter_counts": [6] if args.reference_rainfall else [5, 6, 6],
            "regional_observed_rainfall_reconstruction": args.reference_rainfall,
            "rainfall_caveat": "meteorological observations, no runoff fit; proxy boundaries",
            "parameter_bounds": "unchanged from regional_water_v2 and reach_loss_v1",
            "soil_snow_coefficient": 1,
            "storage_meaning": "consecutive December model-state differences; not observed",
            "accounts": ["legacy", "sector_2022"],
            "selection": "report all; development LOYO relative RMSE first, no selection by 2023",
            "folds": "refit all free parameters; fixed boundary and external storage states",
            "climate": "TerraClimate V1.1 previously chosen on development years; conditional CV",
            "references": [
                "https://www.climatologylab.org/terraclimate.html",
                "https://www.gonghe.gov.cn/lnb/zjgh1/ghnj1__zjgh/ghnj/content_1013653549",
            ],
        },
    )
    sources = [
        Path(__file__),
        output / "protocol.json",
        root / "project/diagnostics/gonghe_connectivity_v1/zones.tif",
        root / "project/diagnostics/gonghe_connectivity_v1/summary.json",
        root / "project/diagnostics/storage_change_v1/storage_changes.csv",
        root / "project/diagnostics/storage_change_v1/summary.json",
        root / "project/calibration/regional_water_v1/manifest.json",
        root / "project/calibration/regional_water_v1/base_tables.json",
        root / "src/ecologyhydro/reach_loss.py",
        root / "src/ecologyhydro/regional_calibration.py",
        root / "src/ecologyhydro/storage_diagnostic.py",
        root / "scripts/calibrate_regional_water.py",
    ]
    if args.reference_rainfall:
        sources.append(root / "project/diagnostics/reference_precipitation_v1/summary.json")
        sources.extend(
            root / f"project/diagnostics/reference_precipitation_v1/precipitation_{year}.tif"
            for year in YEARS
        )
    write_json(output / "manifest.json", {"sources": [fingerprint(p) for p in sources]})
    started, peak = time.perf_counter(), 0
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=worker, args=(p, str(output), args.reference_rainfall))
        for p in ("fine", "copernicus")
    ]
    for process in processes:
        process.start()
    try:
        while any(process.is_alive() for process in processes):
            memory = psutil.Process().memory_info().rss
            for child in psutil.Process().children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    memory += child.memory_info().rss
            peak = max(peak, memory)
            if memory > 24 * 1024**3 or time.perf_counter() - started > 12 * 3600:
                raise RuntimeError("Resource budget exceeded")
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError("Worker failed")
            time.sleep(1)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
    if any(process.exitcode != 0 for process in processes):
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
